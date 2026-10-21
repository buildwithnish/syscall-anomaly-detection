import json
import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------
# Project import setup
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, PROJECT_ROOT)

from src.models.lstm_autoencoder import LSTMAutoencoder


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_FILE = "models/v2_lstm_autoencoder.pth"
CONFIG_FILE = "models/v2_model_config.json"
THRESHOLD_FILE = "models/threshold.json"

BATCH_SIZE = 512


# ---------------------------------------------------------
# Load V2 configuration
# ---------------------------------------------------------

def load_config():

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    return config


# ---------------------------------------------------------
# Map raw syscall IDs to V2 vocabulary indices
# ---------------------------------------------------------

def map_to_vocabulary(sequences, config):

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

        mapped_sequences[
            sequences == syscall_id
        ] = vocab_index

    unknown_mask = ~np.isin(
        sequences,
        known_syscalls
    )

    unknown_events = int(
        unknown_mask.sum()
    )

    return mapped_sequences, unknown_events


# ---------------------------------------------------------
# Score dataset
# ---------------------------------------------------------

def score_dataset(input_file, output_file):

    print()
    print("V2 DATASET SCORING")
    print("=" * 50)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    raw_sequences = np.load(input_file)

    print("Input file         :", input_file)
    print("Loaded sequences   :", len(raw_sequences))
    print("Sequence shape     :", raw_sequences.shape)

    # -----------------------------------------------------
    # Load configuration
    # -----------------------------------------------------

    config = load_config()

    print("Known syscalls     :", len(config["unique_syscalls"]))
    print("UNK index          :", config["unk_index"])
    print("Vocabulary size    :", config["vocab_size"])

    # -----------------------------------------------------
    # Map raw syscall IDs → vocabulary indices
    # -----------------------------------------------------

    sequences, unknown_events = map_to_vocabulary(
        raw_sequences,
        config
    )

    print("Unknown events     :", unknown_events)
    print("Mapped min index   :", sequences.min())
    print("Mapped max index   :", sequences.max())

    # -----------------------------------------------------
    # Load frozen threshold
    # -----------------------------------------------------

    with open(THRESHOLD_FILE, "r") as f:
        threshold_config = json.load(f)

    threshold = threshold_config["threshold"]

    print("Threshold          :", f"{threshold:.6f}")

    # -----------------------------------------------------
    # Load V2 model
    # -----------------------------------------------------

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

    print("Device             : cpu")

    # -----------------------------------------------------
    # Convert to tensor
    # -----------------------------------------------------

    sequences_tensor = torch.tensor(
        sequences,
        dtype=torch.long
    )

    # -----------------------------------------------------
    # Calculate reconstruction scores
    # -----------------------------------------------------

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

            target_probabilities = probabilities.gather(
                2,
                batch.unsqueeze(-1)
            ).squeeze(-1)

            reconstruction_error = -torch.log(
                target_probabilities.clamp(
                    min=1e-9
                )
            ).mean(dim=1)

            scores.extend(
                reconstruction_error.cpu().numpy()
            )

    scores = np.array(
        scores,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Save scores
    # -----------------------------------------------------

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    np.save(
        output_file,
        scores
    )

    # -----------------------------------------------------
    # Detection statistics
    # -----------------------------------------------------

    detected = int(
        np.sum(scores > threshold)
    )

    normal = len(scores) - detected

    detection_rate = (
        detected / len(scores)
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()
    print("RESULTS")
    print("=" * 50)

    print("Total sequences    :", len(scores))
    print("Predicted normal   :", normal)
    print("Predicted anomalies:", detected)

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
    print("Scores saved to    :", output_file)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
        )

        print(
            "python src/detection/score_v2_dataset.py "
            "<input_sequences.npy> "
            "<output_scores.npy>"
        )

        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    score_dataset(
        input_file,
        output_file
    )