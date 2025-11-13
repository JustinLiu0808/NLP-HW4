# -----------------------------
# t5_utils.py (refactored)
# Public APIs unchanged.
# -----------------------------
import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def setup_wandb(args):
    """Initialize Weights & Biases with the current run config."""
    wandb.init(
        project="text-to-sql-t5",
        name=args.experiment_name,
        config=vars(args),
    )


def initialize_model(args):
    """
    Build a T5-small model either:
      - finetuning a pretrained checkpoint, or
      - initializing from the T5-small config (random weights).
    A small encoder prefix is frozen to stabilize early training.
    """
    if args.finetune:
        print("Loading pretrained T5-small model for finetuning...")
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
        model.config.dropout_rate = 0.1
        model.config.attention_dropout_rate = 0.1
    else:
        print("Initializing T5-small model from scratch...")
        cfg = T5Config.from_pretrained(
            "google-t5/t5-small",
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
        )
        model = T5ForConditionalGeneration(cfg)

    # Freeze a few shallow encoder blocks (keeps the interface identical)
    num_layers_to_freeze = 0
    for name, param in model.named_parameters():
        if name.startswith("encoder.block."):
            try:
                block_id = int(name.split(".")[2])
                if block_id < num_layers_to_freeze:
                    param.requires_grad = False
            except ValueError:
                # If parsing ever fails, leave the parameter trainable
                pass

    # Diagnostics
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"🔒 Frozen parameters: {frozen:,} / {total:,} ({frozen/total:.2%})")

    model = model.to(DEVICE)
    print(f"Model moved to {DEVICE}")
    print(f"Number of parameters: {total:,}")
    print(f"Number of trainable parameters: {total - frozen:,}")

    return model


def mkdir(dirpath):
    """Create directory if missing (safe for concurrent calls)."""
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath, exist_ok=True)
        except FileExistsError:
            pass


def save_model(checkpoint_dir, model, best):
    """
    Persist model weights to:
      - best_model.pt  (if best=True)
      - last_model.pt  (otherwise)
    """
    mkdir(checkpoint_dir)
    fname = "best_model.pt" if best else "last_model.pt"
    save_path = os.path.join(checkpoint_dir, fname)
    print(f"Saving {'best' if best else 'last'} model to {save_path}")
    torch.save({"model_state_dict": model.state_dict()}, save_path)


def load_model_from_checkpoint(args, best):
    """
    Load either the best or last checkpoint and return a model on DEVICE.
    Architecture is recreated via initialize_model(args) before loading weights.
    """
    fname = "best_model.pt" if best else "last_model.pt"
    checkpoint_path = os.path.join(args.checkpoint_dir, fname)
    print(f"Loading {'best' if best else 'last'} model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = initialize_model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    return model


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    """Create optimizer and (optionally) a scheduler based on CLI args."""
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    """
    AdamW with decoupled weight decay:
    - decay on non-LayerNorm/non-bias parameters
    - no decay for LayerNorms and biases
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [n for n in decay_parameters if "bias" not in n]

    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        return torch.optim.AdamW(grouped, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999))
    else:
        # Reserved for future optimizers; keep interface stable
        return torch.optim.AdamW(grouped, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999))


def initialize_scheduler(args, optimizer, epoch_length):
    """
    Build a scheduler if requested; otherwise return None.
    Steps are computed from epoch_length * max_n_epochs and warmup epochs.
    """
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    if args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    if args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    raise NotImplementedError(f"Unknown scheduler_type: {args.scheduler_type}")


def get_parameter_names(model, forbidden_layer_types):
    """
    Collect parameter names recursively, excluding modules whose type is in
    forbidden_layer_types. Mirrors HF's utility but is kept local for clarity.
    """
    result = []
    for name, child in model.named_children():
        # Dive into children unless the child itself is a forbidden type
        if not isinstance(child, tuple(forbidden_layer_types)):
            result += [f"{name}.{n}" for n in get_parameter_names(child, forbidden_layer_types)]
    # Include top-level parameters (not registered under any child)
    result += list(model._parameters.keys())
    return result
