import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Project import path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# Configuration
# ============================================================

SEQUENCES_FILE = "data/processed/sequences.npy"
MODEL_FILE = "models/lstm_autoencoder.pth"
CONFIG_FILE = "models/model_config.json"

SCORES_FILE = "data/processed/anomaly_scores.npy"
THRESHOLD_FILE = "models/threshold.json"


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load configuration
# ============================================================

with open(CONFIG_FILE, "r") as file:
    config = json.load(file)


VOCAB_SIZE = config["vocab_size"]
EMBEDDING_DIM = config["embedding_dim"]
HIDDEN_DIM = config["hidden_dim"]
SEQUENCE_LENGTH = config["sequence_length"]

UNK_INDEX = config["unk_index"]
UNIQUE_SYSCALLS = config["unique_syscalls"]


# ============================================================
# Build syscall vocabulary
# ============================================================

SYSCALL_TO_INDEX = {
    int(syscall_id): index
    for index, syscall_id in enumerate(UNIQUE_SYSCALLS)
}


# ============================================================
# Load sequences
# ============================================================

sequences = np.load(SEQUENCES_FILE)

print("=" * 60)
print("ANOMALY SCORING")
print("=" * 60)

print(f"Loaded sequences : {len(sequences)}")
print(f"Sequence shape   : {sequences.shape}")
print(f"Vocabulary size  : {VOCAB_SIZE}")
print(f"UNK index        : {UNK_INDEX}")
print(f"Device           : {DEVICE}")


# ============================================================
# Convert syscall IDs to vocabulary indices
# ============================================================

mapped_sequences = np.empty_like(
    sequences,
    dtype=np.int64
)

unknown_count = 0

for i, sequence in enumerate(sequences):

    for j, syscall_id in enumerate(sequence):

        syscall_id = int(syscall_id)

        if syscall_id in SYSCALL_TO_INDEX:

            mapped_sequences[i, j] = SYSCALL_TO_INDEX[
                syscall_id
            ]

        else:

            mapped_sequences[i, j] = UNK_INDEX
            unknown_count += 1


print(
    f"Unknown syscall events : {unknown_count}"
)


# ============================================================
# Convert to tensor
# ============================================================

data = torch.tensor(
    mapped_sequences,
    dtype=torch.long
).to(DEVICE)


# ============================================================
# Load model
# ============================================================

model = LSTMAutoencoder(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
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
# Calculate anomaly scores
# ============================================================

criterion = nn.CrossEntropyLoss(
    reduction="none"
)

scores = []


with torch.no_grad():

    for start in range(
        0,
        len(data),
        256
    ):

        batch = data[
            start:start + 256
        ]

        outputs = model(batch)

        # ----------------------------------------------------
        # Calculate token-level cross entropy
        # ----------------------------------------------------

        batch_loss = criterion(
            outputs.reshape(
                -1,
                VOCAB_SIZE
            ),
            batch.reshape(-1)
        )

        # ----------------------------------------------------
        # Reshape back to sequences
        # ----------------------------------------------------

        batch_loss = batch_loss.reshape(
            batch.shape[0],
            SEQUENCE_LENGTH
        )

        # ----------------------------------------------------
        # Mean reconstruction error per sequence
        # ----------------------------------------------------

        sequence_scores = batch_loss.mean(
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
# Save anomaly scores
# ============================================================

os.makedirs(
    os.path.dirname(SCORES_FILE),
    exist_ok=True
)

np.save(
    SCORES_FILE,
    scores
)


# ============================================================
# Calculate initial threshold
# ============================================================

threshold = float(
    np.percentile(
        scores,
        95
    )
)


# ============================================================
# Save threshold
# ============================================================

threshold_data = {

    "threshold": threshold,

    "method": "95th_percentile",

    "num_sequences": len(scores),

    "mean_score": float(
        np.mean(scores)
    ),

    "median_score": float(
        np.median(scores)
    ),

    "percentile_95": float(
        np.percentile(scores, 95)
    ),

    "percentile_99": float(
        np.percentile(scores, 99)
    )

}


with open(
    THRESHOLD_FILE,
    "w"
) as file:

    json.dump(
        threshold_data,
        file,
        indent=4
    )


# ============================================================
# Statistics
# ============================================================

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
    f"Median score    : {np.median(scores):.6f}"
)

print(
    f"95th percentile : {np.percentile(scores, 95):.6f}"
)

print(
    f"99th percentile : {np.percentile(scores, 99):.6f}"
)

print(
    f"Threshold       : {threshold:.6f}"
)

print("-" * 60)

print(
    f"Scores saved    : {SCORES_FILE}"
)

print(
    f"Threshold saved : {THRESHOLD_FILE}"
)

print("=" * 60)
print("SCORING COMPLETE")
print("=" * 60)