import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# V3 PID-AWARE DATASET SCORING
# ============================================================

MODEL_FILE = "models/v3_pid_lstm_autoencoder.pth"
CONFIG_FILE = "models/v2_model_config.json"
THRESHOLD_FILE = "models/threshold.json"

BATCH_SIZE = 512


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def map_to_vocabulary(sequences, config):
    """
    Convert raw syscall IDs into the fixed V2/V3 vocabulary.

    Unknown syscall IDs are mapped to <UNK>.
    """

    known_syscalls = config["unique_syscalls"]
    unk_index = config["unk_index"]

    syscall_to_index = {
        syscall_id: index
        for index, syscall_id in enumerate(known_syscalls)
    }

    mapped_sequences = np.full(
        sequences.shape,
        unk_index,
        dtype=np.int64
    )

    for syscall_id, vocab_index in syscall_to_index.items():
        mapped_sequences[sequences == syscall_id] = vocab_index

    unknown_mask = ~np.isin(
        sequences,
        known_syscalls
    )

    unknown_events = int(
        unknown_mask.sum()
    )

    return mapped_sequences, unknown_events


def score_dataset(input_file, output_file):

    print()
    print("V3 PID-AWARE DATASET SCORING")
    print("=" * 55)

    # --------------------------------------------------------
    # Load raw syscall sequences
    # --------------------------------------------------------

    raw_sequences = np.load(input_file)

    print("Input file         :", input_file)
    print("Loaded sequences   :", len(raw_sequences))
    print("Sequence shape     :", raw_sequences.shape)

    # --------------------------------------------------------
    # Load vocabulary configuration
    # --------------------------------------------------------

    config = load_config()

    print("Known syscalls     :", len(config["unique_syscalls"]))
    print("UNK index          :", config["unk_index"])
    print("Vocabulary size    :", config["vocab_size"])

    # --------------------------------------------------------
    # Map raw syscall IDs to vocabulary
    # --------------------------------------------------------

    sequences, unknown_events = map_to_vocabulary(
        raw_sequences,
        config
    )

    print("Unknown events     :", unknown_events)
    print("Mapped min index   :", sequences.min())
    print("Mapped max index   :", sequences.max())

    # --------------------------------------------------------
    # Load frozen threshold
    # --------------------------------------------------------

    with open(THRESHOLD_FILE, "r") as f:
        threshold_config = json.load(f)

    threshold = threshold_config["threshold"]

    print(
        "Threshold          :",
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # Load V3 model
    # --------------------------------------------------------

    model = LSTMAutoencoder(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"]
    )

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location="cpu"
        )
    )

    model.eval()

    print("Model              : V3 PID-aware LSTM")
    print("Device             : cpu")

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    sequences_tensor = torch.tensor(
        sequences,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Calculate reconstruction scores
    # --------------------------------------------------------

    scores = []

    with torch.no_grad():

        for batch in torch.split(
            sequences_tensor,
            BATCH_SIZE
        ):

            output = model(batch)

            probabilities = torch.softmax(
                output,
                dim=-1
            )

            target_probabilities = (
                probabilities
                .gather(
                    2,
                    batch.unsqueeze(-1)
                )
                .squeeze(-1)
            )

            reconstruction_error = (
                -torch.log(
                    target_probabilities.clamp(
                        min=1e-9
                    )
                ).mean(dim=1)
            )

            scores.extend(
                reconstruction_error
                .cpu()
                .numpy()
            )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Save scores
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        output_file
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    np.save(
        output_file,
        scores
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    detected = int(
        np.sum(scores > threshold)
    )

    normal = (
        len(scores) - detected
    )

    detection_rate = (
        detected / len(scores)
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("RESULTS")
    print("=" * 55)

    print(
        "Total sequences    :",
        len(scores)
    )

    print(
        "Predicted normal   :",
        normal
    )

    print(
        "Predicted anomalies:",
        detected
    )

    print(
        "Detection/FPR rate :",
        f"{detection_rate:.4%}"
    )

    print(
        "Min score          :",
        f"{scores.min():.6f}"
    )

    print(
        "Max score          :",
        f"{scores.max():.6f}"
    )

    print(
        "Mean score         :",
        f"{scores.mean():.6f}"
    )

    print(
        "Median score       :",
        f"{np.median(scores):.6f}"
    )

    print(
        "95th percentile    :",
        f"{np.percentile(scores, 95):.6f}"
    )

    print(
        "99th percentile    :",
        f"{np.percentile(scores, 99):.6f}"
    )

    print(
        "Threshold          :",
        f"{threshold:.6f}"
    )

    print()
    print(
        "Scores saved to    :",
        output_file
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage:")
        print(
            "python src/detection/score_v3_dataset.py "
            "<input_sequences.npy> "
            "<output_scores.npy>"
        )

        sys.exit(1)

    score_dataset(
        sys.argv[1],
        sys.argv[2]
    )
