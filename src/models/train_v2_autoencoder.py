import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from lstm_autoencoder import LSTMAutoencoder


SEQUENCES_FILE = "data/processed/v2_normal_sequences_sampled.npy"
CONFIG_FILE = "models/v2_model_config.json"

MODEL_FILE = "models/v2_lstm_autoencoder.pth"
HISTORY_FILE = "models/v2_training_history.json"

BATCH_SIZE = 256
EPOCHS = 8
LEARNING_RATE = 1e-3

UNK_MASK_PROB = 0.05

DEVICE = torch.device("cpu")


class CorruptedSequenceDataset(Dataset):

    def __init__(self, sequences, unk_index):
        self.sequences = torch.tensor(
            sequences,
            dtype=torch.long
        )

        self.unk_index = unk_index

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        clean = self.sequences[index].clone()

        corrupted = clean.clone()

        mask = torch.rand(
            corrupted.shape
        ) < UNK_MASK_PROB

        corrupted[mask] = self.unk_index

        return corrupted, clean


def main():

    print("V2 LSTM AUTOENCODER TRAINING")
    print("=" * 45)

    sequences = np.load(SEQUENCES_FILE)

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]
    unk_index = config["unk_index"]

    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]

    print("Loaded sequences :", len(sequences))
    print("Sequence shape   :", sequences.shape)
    print("Known syscalls   :", len(config["unique_syscalls"]))
    print("UNK index        :", unk_index)
    print("Vocabulary size  :", vocab_size)

    dataset = CorruptedSequenceDataset(
        sequences,
        unk_index
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

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

    print("Training sequences   :", train_size)
    print("Validation sequences :", val_size)
    print("Device               :", DEVICE)

    model = LSTMAutoencoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0
        train_batches = 0

        for corrupted, clean in train_loader:

            corrupted = corrupted.to(DEVICE)
            clean = clean.to(DEVICE)

            optimizer.zero_grad()

            output = model(corrupted)

            loss = criterion(
                output.reshape(-1, vocab_size),
                clean.reshape(-1)
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        model.eval()

        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():

            for corrupted, clean in val_loader:

                corrupted = corrupted.to(DEVICE)
                clean = clean.to(DEVICE)

                output = model(corrupted)

                loss = criterion(
                    output.reshape(-1, vocab_size),
                    clean.reshape(-1)
                )

                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

    print()
    print("Training complete.")
    print("Model saved to   :", MODEL_FILE)
    print("History saved to :", HISTORY_FILE)


if __name__ == "__main__":
    main()
