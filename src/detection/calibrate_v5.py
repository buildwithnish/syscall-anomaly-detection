import json
import numpy as np


INPUT_FILE = "data/processed/normal_calibration_sequences.npy"
OUTPUT_FILE = "models/v5_normalization.json"


# The V5 scorer uses the same behavioral formulas as the
# production scorer. This script calculates normalization
# bounds ONLY from the normal calibration corpus.


ALPHA = 1.0

BEHAVIOR_FILE = "models/v5_behavior_stats.json"


def main():

    print()
    print("V5 NORMALIZATION CALIBRATION")
    print("=" * 60)

    behavior = json.load(
        open(BEHAVIOR_FILE, "r")
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

    sequences = np.load(INPUT_FILE)

    rarity_scores = []
    transition_scores = []

    syscall_denominator = (
        total_events
        + ALPHA * syscall_vocab_size
    )

    transition_denominator = (
        total_transitions
        + ALPHA * transition_vocab_size
    )

    unseen_transition_probability = (
        ALPHA
        / (
            total_transitions
            + ALPHA * (
                transition_vocab_size + 1
            )
        )
    )

    for sequence in sequences:

        rarity = []

        for syscall_id in sequence:

            key = str(int(syscall_id))

            count = syscall_counts.get(
                key,
                0
            )

            probability = (
                count + ALPHA
            ) / syscall_denominator

            rarity.append(
                -np.log(probability)
            )

        rarity_scores.append(
            float(np.mean(rarity))
        )

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
                ) / transition_denominator

            else:

                probability = (
                    unseen_transition_probability
                )

            transitions.append(
                -np.log(probability)
            )

        transition_scores.append(
            float(np.mean(transitions))
        )

    rarity_scores = np.asarray(
        rarity_scores,
        dtype=np.float32
    )

    transition_scores = np.asarray(
        transition_scores,
        dtype=np.float32
    )

    output = {
        "method": "calibration_percentile_bounds",
        "source": INPUT_FILE,
        "alpha": ALPHA,
        "rarity": {
            "lower": float(
                np.percentile(
                    rarity_scores,
                    1
                )
            ),
            "upper": float(
                np.percentile(
                    rarity_scores,
                    99
                )
            )
        },
        "transition": {
            "lower": float(
                np.percentile(
                    transition_scores,
                    1
                )
            ),
            "upper": float(
                np.percentile(
                    transition_scores,
                    99
                )
            )
        }
    }

    with open(
        OUTPUT_FILE,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("Calibration sequences :", len(sequences))

    print(
        "Rarity lower          :",
        f"{output['rarity']['lower']:.6f}"
    )

    print(
        "Rarity upper          :",
        f"{output['rarity']['upper']:.6f}"
    )

    print(
        "Transition lower      :",
        f"{output['transition']['lower']:.6f}"
    )

    print(
        "Transition upper      :",
        f"{output['transition']['upper']:.6f}"
    )

    print()
    print(
        "Saved to              :",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
