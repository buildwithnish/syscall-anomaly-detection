import json
import os
import sys

import numpy as np
import torch

from src.models.lstm_autoencoder import LSTMAutoencoder


MODEL_FILE = "models/v2_lstm_autoencoder.pth"
MODEL_CONFIG_FILE = "models/v2_model_config.json"
BEHAVIOR_FILE = "models/v5_behavior_stats.json"
NORMALIZATION_FILE = "models/v5_normalization.json"

BATCH_SIZE = 512
ALPHA = 1.0


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_raw_to_model_mapping(config):

    unique_syscalls = config["unique_syscalls"]

    raw_to_model = {
        int(syscall_id): index
        for index, syscall_id
        in enumerate(unique_syscalls)
    }

    unk_index = int(
        config.get(
            "unk_index",
            len(unique_syscalls)
        )
    )

    return raw_to_model, unk_index


def map_sequences(
    sequences,
    raw_to_model,
    unk_index
):

    mapped = np.full(
        sequences.shape,
        unk_index,
        dtype=np.int64
    )

    known_events = 0
    unknown_events = 0

    flat_input = sequences.reshape(-1)
    flat_output = mapped.reshape(-1)

    for i, raw_id in enumerate(flat_input):

        raw_id = int(raw_id)

        if raw_id in raw_to_model:

            flat_output[i] = raw_to_model[raw_id]
            known_events += 1

        else:

            flat_output[i] = unk_index
            unknown_events += 1

    return (
        mapped,
        known_events,
        unknown_events
    )


def calculate_reconstruction_scores(
    model,
    sequences,
    device
):

    model.eval()

    scores = []

    with torch.no_grad():

        for start in range(
            0,
            len(sequences),
            BATCH_SIZE
        ):

            batch = sequences[
                start:start + BATCH_SIZE
            ]

            x = torch.tensor(
                batch,
                dtype=torch.long,
                device=device
            )

            output = model(x)

            log_probs = torch.log_softmax(
                output,
                dim=-1
            )

            target_log_probs = (
                log_probs
                .gather(
                    2,
                    x.unsqueeze(-1)
                )
                .squeeze(-1)
            )

            reconstruction_error = (
                -target_log_probs.mean(dim=1)
            )

            scores.extend(
                reconstruction_error
                .cpu()
                .numpy()
                .tolist()
            )

    return np.asarray(
        scores,
        dtype=np.float32
    )


def calculate_rarity_scores(
    raw_sequences,
    syscall_counts,
    total_events,
    vocabulary_size
):

    denominator = (
        total_events
        + ALPHA * vocabulary_size
    )

    scores = []

    for sequence in raw_sequences:

        rarity = []

        for syscall_id in sequence:

            key = str(int(syscall_id))

            count = syscall_counts.get(
                key,
                0
            )

            probability = (
                count + ALPHA
            ) / denominator

            rarity.append(
                -np.log(probability)
            )

        scores.append(
            float(np.mean(rarity))
        )

    return np.asarray(
        scores,
        dtype=np.float32
    )


def calculate_transition_scores(
    raw_sequences,
    transition_counts,
    total_transitions,
    observed_transition_vocab_size
):

    denominator = (
        total_transitions
        + ALPHA * observed_transition_vocab_size
    )

    unseen_probability = (
        ALPHA
        / (
            total_transitions
            + ALPHA * (
                observed_transition_vocab_size + 1
            )
        )
    )

    scores = []

    for sequence in raw_sequences:

        transitions = []

        for i in range(len(sequence) - 1):

            first = str(
                int(sequence[i])
            )

            second = str(
                int(sequence[i + 1])
            )

            key = f"{first},{second}"

            if key in transition_counts:

                count = transition_counts[key]

                probability = (
                    count + ALPHA
                ) / denominator

            else:

                probability = unseen_probability

            transitions.append(
                -np.log(probability)
            )

        if transitions:

            scores.append(
                float(np.mean(transitions))
            )

        else:

            scores.append(0.0)

    return np.asarray(
        scores,
        dtype=np.float32
    )


def normalize(
    values,
    lower,
    upper
):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    denominator = upper - lower

    if denominator <= 0:

        return np.zeros_like(
            values,
            dtype=np.float32
        )

    normalized = (
        values - lower
    ) / denominator

    return np.clip(
        normalized,
        0.0,
        1.0
    )


def main():

    if len(sys.argv) != 3:

        print()
        print("Usage:")
        print(
            "PYTHONPATH=. python "
            "src/detection/score_v5_hybrid.py "
            "<input_sequences.npy> "
            "<output_scores.npy>"
        )

        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print()
    print("V5 HYBRID BEHAVIORAL SCORING")
    print("=" * 60)

    config = load_json(
        MODEL_CONFIG_FILE
    )

    vocab_size = int(
        config["vocab_size"]
    )

    embedding_dim = int(
        config["embedding_dim"]
    )

    hidden_dim = int(
        config["hidden_dim"]
    )

    raw_to_model, unk_index = (
        build_raw_to_model_mapping(
            config
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device             :",
        device
    )

    model = LSTMAutoencoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(device)

    behavior = load_json(
        BEHAVIOR_FILE
    )

    normalization = load_json(
        NORMALIZATION_FILE
    )

    syscall_counts = {
        str(k): int(v)
        for k, v in behavior["syscall_counts"].items()
    }

    transition_counts = {
        str(k): int(v)
        for k, v in behavior["transition_counts"].items()
    }

    total_events = int(
        behavior["total_events"]
    )

    total_transitions = int(
        behavior["total_transitions"]
    )

    syscall_vocab_size = int(
        behavior["unique_syscalls"]
    )

    transition_vocab_size = int(
        behavior["unique_transitions"]
    )

    rarity_lower = float(
        normalization["rarity"]["lower"]
    )

    rarity_upper = float(
        normalization["rarity"]["upper"]
    )

    transition_lower = float(
        normalization["transition"]["lower"]
    )

    transition_upper = float(
        normalization["transition"]["upper"]
    )

    print(
        "Input file         :",
        input_file
    )

    raw_sequences = np.load(
        input_file
    )

    print(
        "Loaded sequences   :",
        len(raw_sequences)
    )

    print(
        "Sequence shape     :",
        raw_sequences.shape
    )

    print(
        "Model vocabulary   :",
        vocab_size
    )

    print(
        "Known vocabulary   :",
        len(raw_to_model)
    )

    print(
        "UNK index          :",
        unk_index
    )

    (
        mapped_sequences,
        known_events,
        unknown_events
    ) = map_sequences(
        raw_sequences,
        raw_to_model,
        unk_index
    )

    print()
    print("VOCABULARY MAPPING")
    print("-" * 60)

    print(
        "Known events       :",
        known_events
    )

    print(
        "Unknown events     :",
        unknown_events
    )

    print()
    print(
        "Calculating reconstruction scores..."
    )

    reconstruction_scores = (
        calculate_reconstruction_scores(
            model,
            mapped_sequences,
            device
        )
    )

    print(
        "Calculating smoothed syscall rarity..."
    )

    rarity_scores = (
        calculate_rarity_scores(
            raw_sequences,
            syscall_counts,
            total_events,
            syscall_vocab_size
        )
    )

    print(
        "Calculating smoothed transition novelty..."
    )

    transition_scores = (
        calculate_transition_scores(
            raw_sequences,
            transition_counts,
            total_transitions,
            transition_vocab_size
        )
    )

    normalized_rarity = normalize(
        rarity_scores,
        rarity_lower,
        rarity_upper
    )

    normalized_transition = normalize(
        transition_scores,
        transition_lower,
        transition_upper
    )

    hybrid_scores = (
        0.70 * reconstruction_scores
        +
        0.15 * normalized_rarity
        +
        0.15 * normalized_transition
    )

    print()
    print("RESULTS")
    print("=" * 60)

    print("Reconstruction weight : 0.70")
    print("Rarity weight         : 0.15")
    print("Transition weight     : 0.15")

    print()
    print(
        "Frozen rarity bounds :",
        f"{rarity_lower:.6f}",
        "to",
        f"{rarity_upper:.6f}"
    )

    print(
        "Frozen transition bounds:",
        f"{transition_lower:.6f}",
        "to",
        f"{transition_upper:.6f}"
    )

    print()
    print(
        "Reconstruction mean  :",
        f"{reconstruction_scores.mean():.6f}"
    )

    print(
        "Rarity mean          :",
        f"{rarity_scores.mean():.6f}"
    )

    print(
        "Transition mean      :",
        f"{transition_scores.mean():.6f}"
    )

    print()
    print(
        "Hybrid min           :",
        f"{hybrid_scores.min():.6f}"
    )

    print(
        "Hybrid max           :",
        f"{hybrid_scores.max():.6f}"
    )

    print(
        "Hybrid mean          :",
        f"{hybrid_scores.mean():.6f}"
    )

    print(
        "Hybrid median        :",
        f"{np.median(hybrid_scores):.6f}"
    )

    print(
        "Hybrid 95th pct      :",
        f"{np.percentile(hybrid_scores, 95):.6f}"
    )

    print(
        "Hybrid 99th pct      :",
        f"{np.percentile(hybrid_scores, 99):.6f}"
    )

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
        hybrid_scores
    )

    print()
    print(
        "Scores saved to      :",
        output_file
    )


if __name__ == "__main__":
    main()
