import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_calibration_sequences.npy"
)

CONFIG_FILE = (
    PROJECT_ROOT
    / "models"
    / "model_config.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "lstm_autoencoder.pth"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_calibration_scores.npy"
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)


VOCAB_SIZE = config["vocab_size"]
UNK_INDEX = config["unk_index"]

UNIQUE_SYSCALLS = set(
    int(x)
    for x in config["unique_syscalls"]
)

SYSCALL_TO_INDEX = {
    syscall_id: index
    for index, syscall_id
    in enumerate(config["unique_syscalls"])
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = LSTMAutoencoder(
    vocab_size=VOCAB_SIZE,
    embedding_dim=config["embedding_dim"],
    hidden_dim=config["hidden_dim"]
).to(DEVICE)


model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# ============================================================
# LOAD DATA
# ============================================================

sequences = np.load(SEQUENCES_FILE)

print("=" * 60)
print("CALIBRATION DATASET SCORING")
print("=" * 60)

print(
    f"Loaded sequences : {len(sequences)}"
)

print(
    f"Sequence shape   : {sequences.shape}"
)

print(
    f"Vocabulary size  : {VOCAB_SIZE}"
)

print(
    f"UNK index        : {UNK_INDEX}"
)

print(
    f"Device           : {DEVICE}"
)


# ============================================================
# MAP SYSCALL IDs
# ============================================================

mapped_sequences = []

unknown_events = 0

for sequence in sequences:

    mapped_sequence = []

    for syscall_id in sequence:

        syscall_id = int(syscall_id)

        if syscall_id in UNIQUE_SYSCALLS:

            mapped_sequence.append(
                SYSCALL_TO_INDEX[syscall_id]
            )

        else:

            mapped_sequence.append(
                UNK_INDEX
            )

            unknown_events += 1

    mapped_sequences.append(
        mapped_sequence
    )


mapped_sequences = np.array(
    mapped_sequences,
    dtype=np.int64
)


print(
    f"Unknown syscall events : "
    f"{unknown_events}"
)


# ============================================================
# SCORE SEQUENCES
# ============================================================

criterion = nn.CrossEntropyLoss(
    reduction="none"
)

scores = []

BATCH_SIZE = 256

for start in range(
    0,
    len(mapped_sequences),
    BATCH_SIZE
):

    batch = torch.tensor(
        mapped_sequences[
            start:start + BATCH_SIZE
        ],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        outputs = model(batch)

        batch_size = batch.shape[0]

        sequence_length = batch.shape[1]

        loss = criterion(
            outputs.reshape(
                -1,
                VOCAB_SIZE
            ),
            batch.reshape(-1)
        )

        loss = loss.reshape(
            batch_size,
            sequence_length
        )

        sequence_scores = loss.mean(
            dim=1
        )

        scores.extend(
            sequence_scores.cpu().numpy()
        )


scores = np.array(
    scores,
    dtype=np.float32
)


# ============================================================
# STATISTICS
# ============================================================

p50 = np.percentile(scores, 50)
p90 = np.percentile(scores, 90)
p95 = np.percentile(scores, 95)
p97 = np.percentile(scores, 97)
p98 = np.percentile(scores, 98)
p99 = np.percentile(scores, 99)
p995 = np.percentile(scores, 99.5)
p999 = np.percentile(scores, 99.9)


print()
print("-" * 60)

print(
    f"Total sequences : {len(scores)}"
)

print(
    f"Min score       : {scores.min():.6f}"
)

print(
    f"Max score       : {scores.max():.6f}"
)

print(
    f"Mean score      : {scores.mean():.6f}"
)

print(
    f"Median score    : {p50:.6f}"
)

print(
    f"90th percentile : {p90:.6f}"
)

print(
    f"95th percentile : {p95:.6f}"
)

print(
    f"97th percentile : {p97:.6f}"
)

print(
    f"98th percentile : {p98:.6f}"
)

print(
    f"99th percentile : {p99:.6f}"
)

print(
    f"99.5th percentile : {p995:.6f}"
)

print(
    f"99.9th percentile : {p999:.6f}"
)


# ============================================================
# SAVE SCORES
# ============================================================

np.save(
    OUTPUT_FILE,
    scores
)


print("-" * 60)

print(
    f"Scores saved : {OUTPUT_FILE}"
)

print("=" * 60)
print("CALIBRATION SCORING COMPLETE")
print("=" * 60)
