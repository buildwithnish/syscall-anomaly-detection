import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lstm_autoencoder import LSTMAutoencoder

# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCES_FILE = "data/processed/sequences.npy"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_autoencoder.pth")
CONFIG_FILE = os.path.join(MODEL_DIR, "model_config.json")

EMBEDDING_DIM = 32
HIDDEN_DIM = 64

BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001

# During training, randomly replace some input syscalls
# with <UNK>. The original syscall remains the target.
UNK_MASK_PROB = 0.05

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# DATASET
# ============================================================

class CorruptedSequenceDataset(Dataset):
    """
    Training dataset that randomly replaces input tokens
    with <UNK>, while keeping the original sequence as
    the reconstruction target.

    Example:

        Original:
        [openat, read, close, mmap, ...]

        Model input:
        [openat, <UNK>, close, mmap, ...]

        Target:
        [openat, read, close, mmap, ...]
    """

    def __init__(self, data, unk_index, corruption_probability):
        self.data = data
        self.unk_index = unk_index
        self.corruption_probability = corruption_probability

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        target = self.data[index]

        # Clone so the original training data is not modified
        corrupted_input = target.clone()

        # Random mask
        mask = torch.rand(
            corrupted_input.shape
        ) < self.corruption_probability

        # Replace selected syscall IDs with <UNK>
        corrupted_input[mask] = self.unk_index

        return corrupted_input, target


class CleanSequenceDataset(Dataset):
    """
    Validation dataset.

    Validation uses clean sequences so that validation loss
    measures normal reconstruction performance.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sequence = self.data[index]

        return sequence, sequence


# ============================================================
# MODEL
# ============================================================

class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):

        embedded = self.embedding(x)

        _, (hidden, cell) = self.encoder(embedded)

        decoded, _ = self.decoder(
            embedded,
            (hidden, cell)
        )

        output = self.output_layer(decoded)

        return output


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LSTM AUTOENCODER TRAINING")
print("=" * 70)

sequences = np.load(SEQUENCES_FILE)

print(f"Loaded sequences : {len(sequences)}")
print(f"Sequence shape   : {sequences.shape}")


# ============================================================
# BUILD SYSCALL VOCABULARY
# ============================================================

unique_syscalls = sorted(
    set(
        int(syscall_id)
        for sequence in sequences
        for syscall_id in sequence
    )
)

syscall_to_index = {
    syscall_id: index
    for index, syscall_id in enumerate(unique_syscalls)
}

# Reserve the final index for unknown syscalls
unk_index = len(unique_syscalls)

vocab_size = unk_index + 1


print(
    f"Known syscall vocabulary : {len(unique_syscalls)}"
)

print(
    f"UNK index                : {unk_index}"
)

print(
    f"Total vocabulary size    : {vocab_size}"
)


# ============================================================
# CONVERT RAW SYSCALL IDs TO MODEL INDICES
# ============================================================

mapped_sequences = []

unknown_events = 0

for sequence in sequences:

    mapped_sequence = []

    for syscall_id in sequence:

        syscall_id = int(syscall_id)

        if syscall_id in syscall_to_index:

            mapped_sequence.append(
                syscall_to_index[syscall_id]
            )

        else:

            mapped_sequence.append(unk_index)

            unknown_events += 1

    mapped_sequences.append(mapped_sequence)


mapped_sequences = np.array(
    mapped_sequences,
    dtype=np.int64
)


print(
    f"Unknown events in original training data: "
    f"{unknown_events}"
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

num_sequences = len(mapped_sequences)

split_index = int(num_sequences * 0.8)

train_sequences = mapped_sequences[
    :split_index
]

validation_sequences = mapped_sequences[
    split_index:
]


print(
    f"Training sequences   : "
    f"{len(train_sequences)}"
)

print(
    f"Validation sequences : "
    f"{len(validation_sequences)}"
)


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_tensor = torch.tensor(
    train_sequences,
    dtype=torch.long
)

validation_tensor = torch.tensor(
    validation_sequences,
    dtype=torch.long
)


train_dataset = CorruptedSequenceDataset(
    train_tensor,
    unk_index,
    UNK_MASK_PROB
)

validation_dataset = CleanSequenceDataset(
    validation_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


# ============================================================
# MODEL
# ============================================================

model = LSTMAutoencoder(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

training_losses = []
validation_losses = []


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    total_validation_loss = 0.0

    with torch.no_grad():

        for inputs, targets in validation_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            total_validation_loss += loss.item()


    average_validation_loss = (
        total_validation_loss /
        len(validation_loader)
    )


    training_losses.append(
        average_train_loss
    )

    validation_losses.append(
        average_validation_loss
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {average_train_loss:.6f} | "
        f"Val Loss: {average_validation_loss:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


torch.save(
    model.state_dict(),
    MODEL_FILE
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "architecture": "LSTM Autoencoder",

    "embedding_dim": EMBEDDING_DIM,

    "hidden_dim": HIDDEN_DIM,

    "sequence_length": int(
        sequences.shape[1]
    ),

    "vocab_size": vocab_size,

    "unk_index": unk_index,

    "unk_token": "<UNK>",

    "unique_syscalls": unique_syscalls,

    "batch_size": BATCH_SIZE,

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "unk_mask_probability": UNK_MASK_PROB,

    "corruption_strategy":
        "random_input_token_masking",

    "training_losses":
        training_losses,

    "validation_losses":
        validation_losses
}


with open(
    CONFIG_FILE,
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Model saved : {MODEL_FILE}"
)

print(
    f"Config saved : {CONFIG_FILE}"
)

print(
    f"Vocabulary : "
    f"{len(unique_syscalls)} syscalls + <UNK>"
)

print(
    f"UNK mask probability : "
    f"{UNK_MASK_PROB * 100:.1f}%"
)

print(
    f"Final train loss : "
    f"{training_losses[-1]:.6f}"
)

print(
    f"Final val loss   : "
    f"{validation_losses[-1]:.6f}"
)

print("=" * 70)