# -----------------------------
# load_data.py  (refactored)
# Keep public function names and return values unchanged.
# -----------------------------
import os
import random
import re
import string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# punkt may already exist in the environment; download quietly only if missing
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

import torch
from transformers import T5TokenizerFast

# ---------- constants & shared resources ----------
PAD_IDX = 0
# Initialize the tokenizer once (faster and avoids repeated I/O in collate)
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
# A lightweight prefix to normalize NL→SQL prompting
PROMPT_PREFIX = "Translate the question to SQL: "
# T5 has no explicit BOS; reuse an extra-id token as a start token for decoder
BOS_ID = TOKENIZER.convert_tokens_to_ids("<extra_id_0>")
MAX_LEN = 512


class T5Dataset(Dataset):
    """
    Convert NL/SQL text into tensors consumable by T5.
    - train/dev: provide both encoder and decoder tensors
    - test: provide only encoder tensors (no labels)
    """

    def __init__(self, data_folder, split):
        self.split = split
        self.data_folder = data_folder
        self.tokenizer = TOKENIZER
        self.max_length = MAX_LEN
        self.bos_token_id = BOS_ID

        self.data = self.process_data(data_folder, split, self.tokenizer)

    def _read_lines(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f.readlines()]

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_queries = self._read_lines(nl_path)

        processed = []

        if split == "test":
            # Test split has no SQL labels: build encoder-only samples
            for nl in nl_queries:
                enc = tokenizer(
                    PROMPT_PREFIX + nl,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                processed.append(
                    {
                        "encoder_input_ids": enc["input_ids"].squeeze(0),
                        "encoder_attention_mask": enc["attention_mask"].squeeze(0),
                        "decoder_input_ids": None,
                        "decoder_targets": None,
                    }
                )
            return processed

        # Train/dev: pair each NL with its SQL
        sql_path = os.path.join(data_folder, f"{split}.sql")
        sql_queries = self._read_lines(sql_path)

        assert len(nl_queries) == len(
            sql_queries
        ), f"Mismatch between NL and SQL: {len(nl_queries)} vs {len(sql_queries)}"

        for nl, sql in zip(nl_queries, sql_queries):
            enc = tokenizer(
                PROMPT_PREFIX + nl,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            dec = tokenizer(
                sql,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            processed.append(
                {
                    "encoder_input_ids": enc["input_ids"].squeeze(0),
                    "encoder_attention_mask": enc["attention_mask"].squeeze(0),
                    "decoder_input_ids": dec["input_ids"].squeeze(0),
                    "decoder_targets": dec["input_ids"].squeeze(0),
                }
            )

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    """
    Dynamic padding for train/dev.
    Returns:
      - encoder_ids: B x T
      - encoder_mask: B x T
      - decoder_inputs: B x T'
      - decoder_targets: B x T'
      - initial_decoder_inputs: B x 1 (used for generation/inference)
    """
    enc_ids = [b["encoder_input_ids"] for b in batch]
    enc_mask = [b["encoder_attention_mask"] for b in batch]
    dec_ids_list = [b["decoder_input_ids"] for b in batch]

    # Pad encoder side
    encoder_ids = pad_sequence(enc_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(enc_mask, batch_first=True, padding_value=0)

    # Decoder side: shift-right with BOS; targets are the original tokens
    decoder_inputs_list, decoder_targets_list, initial_list = [], [], []
    for dec in dec_ids_list:
        # Defensive fallback: if something is None, keep shapes valid
        if dec is None:
            dec = torch.tensor([BOS_ID], dtype=torch.long)
        dec_in = torch.cat([torch.tensor([BOS_ID], dtype=torch.long), dec[:-1]])
        decoder_inputs_list.append(dec_in)
        decoder_targets_list.append(dec)
        initial_list.append(torch.tensor([BOS_ID], dtype=torch.long))

    decoder_inputs = pad_sequence(
        decoder_inputs_list, batch_first=True, padding_value=PAD_IDX
    )
    decoder_targets = pad_sequence(
        decoder_targets_list, batch_first=True, padding_value=PAD_IDX
    )
    initial_decoder_inputs = torch.stack(initial_list, dim=0)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    """
    Dynamic padding for test (no targets).
    Returns:
      - encoder_ids: B x T
      - encoder_mask: B x T
      - initial_decoder_inputs: B x 1
    """
    enc_ids = [b["encoder_input_ids"] for b in batch]
    enc_mask = [b["encoder_attention_mask"] for b in batch]

    encoder_ids = pad_sequence(enc_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(enc_mask, batch_first=True, padding_value=0)

    # Each sample starts generation with a single BOS token
    initial_decoder_inputs = torch.full((len(batch), 1), BOS_ID, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = "data"
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def load_prompting_data(data_folder):
    # Train
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    # Dev
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    # Test (no labels)
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
