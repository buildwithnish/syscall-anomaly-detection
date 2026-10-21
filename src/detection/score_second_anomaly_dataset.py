import json
import numpy as np
import torch

from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "models")
)

from lstm_autoencoder import LSTMAutoencoder


SEQUENCE_FILE = "data/processed/anomaly_second_sequences.npy"
CONFIG_FILE = "models/model_config.json"
MODEL_FILE = "models/lstm_autoencoder.pth"
THRESHOLD_FILE = "models/threshold.json"
OUTPUT_FILE = "data/processed/anomaly_second_scores.npy"


print("=" * 60)
print("SECOND ISOLATED ANOMALY DATASET SCORING")
print("=" * 60)


# ------------------------------------------------------------
# Load data and configuration
# ------------------------------------------------------------

sequences = np.load(SEQUENCE_FILE)

with open(CONFIG_FILE, "r") as file:
    config = json.load(file)

with open(THRESHOLD_FILE, "r") as file:
    threshold_config = json.load(file)


vocab_size = config["vocab_size"]
embedding_dim = config["embedding_dim"]
hidden_dim = config["hidden_dim"]
unk_index = config["unk_index"]

threshold = threshold_config["threshold"]

known_syscalls = set(config["unique_syscalls"])


print(f"Loaded sequences : {len(sequences)}")
print(f"Sequence shape   : {sequences.shape}")
print(f"Vocabulary size  : {vocab_size}")
print(f"UNK index        : {unk_index}")
print(f"Threshold        : {threshold:.6f}")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

device = torch.device("cpu")

model = LSTMAutoencoder(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()

print(f"Device           : {device}")


# ------------------------------------------------------------
# Map raw syscall IDs to model vocabulary indices
# ------------------------------------------------------------

syscall_to_index = {
    syscall_id: index
    for index, syscall_id
    in enumerate(config["unique_syscalls"])
}


# Start every value as <UNK>
mapped_sequences = np.full(
    sequences.shape,
    unk_index,
    dtype=np.int64
)


# Replace known raw syscall IDs with vocabulary indices
for syscall_id, vocab_index in syscall_to_index.items():
    mapped_sequences[sequences == syscall_id] = vocab_index


# Count raw syscall events unknown to training vocabulary
unknown_mask = ~np.isin(
    sequences,
    list(known_syscalls)
)

unknown_events = int(unknown_mask.sum())


print(
    f"Unknown syscall events : {unknown_events}"
)


# ------------------------------------------------------------
# Convert to PyTorch tensor
# ------------------------------------------------------------

x = torch.tensor(
    mapped_sequences,
    dtype=torch.long,
    device=device
)


# ------------------------------------------------------------
# Calculate reconstruction loss
# ------------------------------------------------------------

criterion = torch.nn.CrossEntropyLoss(
    reduction="none"
)


scores = []

batch_size = 256


with torch.no_grad():

    for start in range(0, len(x), batch_size):

        batch = x[start:start + batch_size]

        output = model(batch)

        loss = criterion(
            output.reshape(-1, vocab_size),
            batch.reshape(-1)
        )

        loss = loss.reshape(
            batch.size(0),
            batch.size(1)
        )

        # Mean reconstruction loss for each sequence
        sequence_scores = loss.mean(dim=1)

        scores.append(
            sequence_scores.cpu().numpy()
        )


scores = np.concatenate(scores)


# ------------------------------------------------------------
# Apply frozen anomaly threshold
# ------------------------------------------------------------

predictions = scores >= threshold

predicted_anomalies = int(
    predictions.sum()
)

predicted_normal = int(
    (~predictions).sum()
)


detection_rate = (
    predicted_anomalies / len(scores)
)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print()
print("-" * 60)

print(f"Total sequences    : {len(scores)}")
print(f"Predicted normal   : {predicted_normal}")
print(f"Predicted anomalies: {predicted_anomalies}")
print(f"Detection rate     : {detection_rate:.4%}")

print(f"Min score          : {scores.min():.6f}")
print(f"Max score          : {scores.max():.6f}")
print(f"Mean score         : {scores.mean():.6f}")
print(f"Median score       : {np.median(scores):.6f}")
print(
    f"95th percentile    : "
    f"{np.percentile(scores, 95):.6f}"
)
print(
    f"99th percentile    : "
    f"{np.percentile(scores, 99):.6f}"
)

print(f"Threshold          : {threshold:.6f}")

print("-" * 60)


# ------------------------------------------------------------
# Save anomaly scores
# ------------------------------------------------------------

np.save(
    OUTPUT_FILE,
    scores
)


print()
print(f"Scores saved       : {OUTPUT_FILE}")
print("=" * 60)
print("SECOND ISOLATED ANOMALY SCORING COMPLETE")
print("=" * 60)