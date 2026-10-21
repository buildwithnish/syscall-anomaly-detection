import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# V3 PID-AWARE LSTM AUTOENCODER TRAINING
# ============================================================

SEQUENCES_FILE = (
    "data/processed/v3_pid_sequences_mapped.npy"
)

CONFIG_FILE = "models/v2_model_config.json"

MODEL_FILE = "models/v3_pid_lstm_autoencoder.pth"

HISTORY_FILE = "models/v3_pid_training_history.json"

SEQUENCE_LENGTH = 20

EPOCHS = 8

BATCH_SIZE = 256

LEARNING_RATE = 0.001

MASK_PROBABILITY = 0.05

TRAIN_SPLIT = 0.8

RANDOM_SEED = 42


def main():

    print()
    print("V3 PID-AWARE LSTM AUTOENCODER TRAINING")
    print("=" * 60)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    unk_index = config["unk_index"]

    # --------------------------------------------------------
    # Load sequences
    # --------------------------------------------------------

    sequences = np.load(SEQUENCES_FILE)

    print("Loaded sequences :", len(sequences))
    print("Sequence shape   :", sequences.shape)
    print("Known syscalls   :", len(config["unique_syscalls"]))
    print("UNK index        :", unk_index)
    print("Vocabulary size  :", vocab_size)

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    x = torch.tensor(
        sequences,
        dtype=torch.long
    )

    dataset = TensorDataset(x)

    train_size = int(
        len(dataset) * TRAIN_SPLIT
    )

    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print("Training sequences   :", len(train_dataset))
    print("Validation sequences :", len(val_dataset))

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cpu")

    print("Device:", device)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LSTMAutoencoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": []
    }

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0
        train_batches = 0

        for (batch,) in train_loader:

            batch = batch.to(device)

            # ------------------------------------------------
            # Corrupt input using <UNK> masking
            # ------------------------------------------------

            corrupted = batch.clone()

            mask = (
                torch.rand(
                    corrupted.shape,
                    device=device
                ) < MASK_PROBABILITY
            )

            corrupted[mask] = unk_index

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            optimizer.zero_grad()

            output = model(corrupted)

            # ------------------------------------------------
            # Reconstruction loss
            # ------------------------------------------------

            loss = criterion(
                output.reshape(-1, vocab_size),
                batch.reshape(-1)
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        train_loss = (
            total_train_loss / train_batches
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        total_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():

            for (batch,) in val_loader:

                batch = batch.to(device)

                output = model(batch)

                loss = criterion(
                    output.reshape(-1, vocab_size),
                    batch.reshape(-1)
                )

                total_val_loss += loss.item()
                val_batches += 1

        val_loss = (
            total_val_loss / val_batches
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_FILE),
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    with open(HISTORY_FILE, "w") as f:
        json.dump(
            history,
            f,
            indent=2
        )

    print()
    print("Training complete.")
    print("Model saved to   :", MODEL_FILE)
    print("History saved to :", HISTORY_FILE)


if __name__ == "__main__":
    main()
