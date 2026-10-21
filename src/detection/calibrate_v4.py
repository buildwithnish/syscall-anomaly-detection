import json
import os

import numpy as np


# ============================================================
# V4 HYBRID SCORE CALIBRATION
# ============================================================

INPUT_FILE = "data/raw/v2_normal/normal_v2.jsonl"

OUTPUT_FILE = "models/v4_normalization.json"


def percentile_bounds(values):

    lower = float(np.percentile(values, 1))
    upper = float(np.percentile(values, 99))

    return lower, upper


def main():

    print()
    print("V4 NORMALIZATION CALIBRATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load V4 normal calibration components
    #
    # These are calculated directly from the normal corpus.
    # --------------------------------------------------------

    syscall_counts = {}
    transition_counts = {}

    total_events = 0
    total_transitions = 0

    previous_by_pid = {}

    print("Reading normal calibration corpus...")
    print("Input file:", INPUT_FILE)

    with open(INPUT_FILE, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                event = json.loads(line)

                pid = event.get("pid")
                syscall_id = event.get("syscall_id")

                if pid is None or syscall_id is None:
                    continue

                pid = int(pid)
                syscall_id = int(syscall_id)

                syscall_counts[syscall_id] = (
                    syscall_counts.get(
                        syscall_id,
                        0
                    ) + 1
                )

                total_events += 1

                if pid in previous_by_pid:

                    previous = previous_by_pid[pid]

                    key = (
                        previous,
                        syscall_id
                    )

                    transition_counts[key] = (
                        transition_counts.get(
                            key,
                            0
                        ) + 1
                    )

                    total_transitions += 1

                previous_by_pid[pid] = syscall_id

            except (
                json.JSONDecodeError,
                ValueError,
                TypeError
            ):
                continue

    # --------------------------------------------------------
    # Convert counts to probabilities
    # --------------------------------------------------------

    syscall_probabilities = {
        syscall_id: count / total_events
        for syscall_id, count
        in syscall_counts.items()
    }

    transition_probabilities = {
        transition: count / total_transitions
        for transition, count
        in transition_counts.items()
    }

    # --------------------------------------------------------
    # Build calibration sequences
    # --------------------------------------------------------

    # We use the same normal corpus, but only enough
    # information to establish robust normalization bounds.

    rarity_values = []
    transition_values = []

    sequence_length = 20

    pid_events = {}

    # Re-read the corpus and construct PID sequences.

    with open(INPUT_FILE, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                event = json.loads(line)

                pid = event.get("pid")
                syscall_id = event.get("syscall_id")

                if pid is None or syscall_id is None:
                    continue

                pid = int(pid)
                syscall_id = int(syscall_id)

                if pid not in pid_events:
                    pid_events[pid] = []

                pid_events[pid].append(
                    syscall_id
                )

            except (
                json.JSONDecodeError,
                ValueError,
                TypeError
            ):
                continue

    # --------------------------------------------------------
    # Calculate calibration component scores
    # --------------------------------------------------------

    for syscall_sequence in pid_events.values():

        if len(syscall_sequence) < sequence_length:
            continue

        for start in range(
            0,
            len(syscall_sequence)
            - sequence_length
            + 1,
            8
        ):

            sequence = syscall_sequence[
                start:start + sequence_length
            ]

            rarity = []

            for syscall_id in sequence:

                probability = syscall_probabilities.get(
                    syscall_id,
                    1e-9
                )

                rarity.append(
                    -np.log(
                        max(
                            probability,
                            1e-9
                        )
                    )
                )

            rarity_values.append(
                float(np.mean(rarity))
            )

            transitions = []

            for i in range(
                len(sequence) - 1
            ):

                transition = (
                    sequence[i],
                    sequence[i + 1]
                )

                probability = (
                    transition_probabilities.get(
                        transition,
                        1e-9
                    )
                )

                transitions.append(
                    -np.log(
                        max(
                            probability,
                            1e-9
                        )
                    )
                )

            if transitions:

                transition_values.append(
                    float(
                        np.mean(transitions)
                    )
                )

    rarity_values = np.asarray(
        rarity_values,
        dtype=np.float32
    )

    transition_values = np.asarray(
        transition_values,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Calculate robust normalization bounds
    # --------------------------------------------------------

    rarity_lower, rarity_upper = (
        percentile_bounds(
            rarity_values
        )
    )

    transition_lower, transition_upper = (
        percentile_bounds(
            transition_values
        )
    )

    # --------------------------------------------------------
    # Save frozen parameters
    # --------------------------------------------------------

    normalization = {

        "method": "normal_1st_99th_percentile",

        "source": INPUT_FILE,

        "sequence_length": sequence_length,

        "rarity": {
            "lower": rarity_lower,
            "upper": rarity_upper
        },

        "transition": {
            "lower": transition_lower,
            "upper": transition_upper
        },

        "weights": {
            "reconstruction": 0.70,
            "rarity": 0.15,
            "transition": 0.15
        }
    }

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w"
    ) as f:

        json.dump(
            normalization,
            f,
            indent=2
        )

    print()
    print("RESULT")
    print("=" * 60)

    print(
        "Calibration events :",
        total_events
    )

    print(
        "Unique syscalls    :",
        len(syscall_counts)
    )

    print(
        "Unique transitions :",
        len(transition_counts)
    )

    print()
    print(
        "Rarity lower       :",
        f"{rarity_lower:.6f}"
    )

    print(
        "Rarity upper       :",
        f"{rarity_upper:.6f}"
    )

    print(
        "Transition lower   :",
        f"{transition_lower:.6f}"
    )

    print(
        "Transition upper   :",
        f"{transition_upper:.6f}"
    )

    print()
    print(
        "Saved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
