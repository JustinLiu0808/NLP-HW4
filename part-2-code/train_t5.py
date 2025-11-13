# -----------------------------
# train_t5.py (refactored & fixed)
# Public function names and CLI args remain unchanged.
# -----------------------------
import os
import argparse
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import numpy as np
import wandb

from transformers import GenerationConfig, T5TokenizerFast

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

# ---------- device & shared constants ----------
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0

# Initialize tokenizer once (avoid repeated I/O in eval/test)
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
BOS_ID = TOKENIZER.convert_tokens_to_ids("<extra_id_0>")


def get_args():
    """
    CLI arguments for training/finetuning and evaluation phases.
    Keep names and defaults unchanged to remain compatible with external scripts.
    """
    parser = argparse.ArgumentParser(description="T5 training loop")

    # Model hyperparameters
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune T5 or not")

    # Training hyperparameters
    parser.add_argument(
        "--optimizer_type", type=str, default="AdamW", choices=["AdamW"], help="Optimizer choice"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help="LR scheduler type",
    )
    parser.add_argument(
        "--num_warmup_epochs",
        type=int,
        default=0,
        help="Warmup epochs for LR scheduler (if enabled)",
    )
    parser.add_argument("--max_n_epochs", type=int, default=0, help="Total training epochs")
    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=0,
        help="Early stopping patience measured in epochs (dev metric stalls)",
    )

    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--experiment_name", type=str, default="experiment", help="Experiment run name"
    )

    # Data hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    return parser.parse_args()


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    """
    Standard NLL training loop with early stopping.
    Saves both last and best checkpoints into:
      checkpoints/<ft_or_scr>_experiments/<experiment_name>/
    Also exports dev predictions each epoch to {results,records}/t5_*_<exp>_dev.{sql,pkl}.
    For Extra Credit (from-scratch training), export names follow the required 'ft_experiment_ec'.
    """
    best_f1 = -1.0
    epochs_since_improvement = 0

    # Checkpoint directory still reflects *actual* training mode (ft or scr)
    model_type_ckpt = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join("checkpoints", f"{model_type_ckpt}_experiments", args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # ---- Export naming policy ----
    # EC mode = from-scratch training (no --finetune): filenames must be t5_ft_experiment_ec_*.*
    ec_mode = not args.finetune
    export_prefix = "ft"  # required by assignment for EC; fine for finetune as well
    export_experiment_name = "ft_experiment_ec" if ec_mode else "ft_experiment"

    gt_sql_path = os.path.join("data/dev.sql")
    gt_record_path = os.path.join("records/ground_truth_dev.pkl")

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        # export paths for DEV this epoch
        model_sql_path = os.path.join(f"results/t5_{export_prefix}_{export_experiment_name}_dev.sql")
        model_record_path = os.path.join(f"records/t5_{export_prefix}_{export_experiment_name}_dev.pkl")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args,
            model,
            dev_loader,
            gt_sql_path,
            model_sql_path,
            gt_record_path,
            model_record_path,
        )
        print(
            f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, "
            f"Record EM: {record_em}, SQL EM: {sql_em}"
        )
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            wandb.log(
                {
                    "train/loss": tr_loss,
                    "dev/loss": eval_loss,
                    "dev/record_f1": record_f1,
                    "dev/record_em": record_em,
                    "dev/sql_em": sql_em,
                    "dev/error_rate": error_rate,
                },
                step=epoch,
            )

        # Track the best model by Dev Record F1
        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model, best=True)
            print(f"✅ New best model saved (Dev F1={best_f1:.4f})")
        else:
            epochs_since_improvement += 1

        # Always persist the last checkpoint for resuming
        save_model(checkpoint_dir, model, best=False)

        # Keep a fresh copy of best after each improvement
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            print("⏸️ Early stopping triggered.")
            break



def train_epoch(args, model, train_loader, optimizer, scheduler):
    """
    One full pass over the training set.
    Uses label-smoothing CE; masks out PAD targets when computing loss.
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )["logits"]

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def _gen_config():
    """
    Centralized generation config to keep eval/test consistent.
    """
    return GenerationConfig(
        max_length=300,
        num_beams=8,            # default beams for dev
        early_stopping=True,
        length_penalty=0.95,
        decoder_start_token_id=BOS_ID,
    )


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    """
    Dev evaluation:
      - compute token-level loss (masked by PAD)
      - generate SQL strings and save them + execution records
      - compute metrics (SQL EM / Record EM / Record F1) and error rate
    Returns:
      avg_loss, record_f1, record_em, sql_em, error_rate
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    generated_queries = []

    gen_cfg = _gen_config()

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(
            dev_loader, desc="Evaluating"
        ):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # Loss on teacher-forced targets
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

            ntoks = torch.sum(non_pad).item()
            total_loss += loss.item() * ntoks
            total_tokens += ntoks

            # Autoregressive generation (uses BOS and beam search)
            gen_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                **gen_cfg.to_dict(),   # <-- only one source of kwargs
            )

            # Decode into plain SQL strings
            for seq in gen_ids:
                generated_queries.append(TOKENIZER.decode(seq, skip_special_tokens=True))

    avg_loss = total_loss / max(total_tokens, 1)

    # Persist generated SQL and corresponding execution records
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

    # Compute metrics and error rate
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    num_errors = sum(1 for msg in error_msgs if msg != "")
    error_rate = num_errors / len(error_msgs) if error_msgs else 0.0

    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """
    Test-time generation:
      - generate SQL strings only (no labels)
      - save SQL and records and report SQL error rate
    """
    model.eval()

    generated_queries = []
    # Start from base config and override in the dict to avoid duplicate kwargs.
    base_cfg = _gen_config().to_dict()
    base_cfg["num_beams"] = 5  # use a slightly narrower beam at test if desired

    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Testing"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            gen_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                **base_cfg,           # <-- single expanded kwargs dict (no duplicates)
            )
            for seq in gen_ids:
                generated_queries.append(TOKENIZER.decode(seq, skip_special_tokens=True))

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

    # Derive error rate from saved records
    with open(model_record_path, "rb") as f:
        _, error_msgs = pickle.load(f)
    num_errors = sum(1 for msg in error_msgs if msg != "")
    error_rate = num_errors / len(error_msgs) if error_msgs else 0.0

    print(f"Test set: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
    print(f"Generated {len(generated_queries)} test queries")
    print(f"Saved to {model_sql_path} and {model_record_path}")


def main():
    # Parse args and optionally enable W&B
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Data loaders
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)

    # Try resuming from last checkpoint; otherwise start fresh
    checkpoint_path = os.path.join(
        "checkpoints",
        "ft_experiments" if args.finetune else "scr_experiments",
        args.experiment_name,
        "last_model.pt",
    )
    if os.path.exists(checkpoint_path):
        print(f"🧩 Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model = initialize_model(args)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("🚀 Starting training from scratch..." if not args.finetune else "🚀 Starting finetuning...")
        model = initialize_model(args)

    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train / finetune
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Load best checkpoint for evaluation and export
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    # ---- Export naming policy ----
    ec_mode = not args.finetune
    export_prefix = "ft"  # required by assignment for EC; fine for finetune as well
    export_experiment_name = "ft_experiment_ec" if ec_mode else "ft_experiment"

    # --- Dev evaluation and export ---
    gt_sql_path = os.path.join("data/dev.sql")
    gt_record_path = os.path.join("records/ground_truth_dev.pkl")
    dev_sql_path = os.path.join(f"results/t5_{export_prefix}_{export_experiment_name}_dev.sql")
    dev_record_path = os.path.join(f"records/t5_{export_prefix}_{export_experiment_name}_dev.pkl")

    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, dev_sql_path, gt_record_path, dev_record_path
    )
    print(
        f"Dev set results: Loss: {dev_loss:.4f}, Record F1: {dev_record_f1:.4f}, "
        f"Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}"
    )
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # --- Test generation and export ---
    test_sql_path = os.path.join(f"results/t5_{export_prefix}_{export_experiment_name}_test.sql")
    test_record_path = os.path.join(f"records/t5_{export_prefix}_{export_experiment_name}_test.pkl")
    test_inference(args, model, test_loader, test_sql_path, test_record_path)



if __name__ == "__main__":
    main()
