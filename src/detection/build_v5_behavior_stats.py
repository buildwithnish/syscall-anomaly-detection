import json
from collections import Counter, defaultdict


INPUT_FILE = "data/raw/v2_normal/normal_v2.jsonl"
OUTPUT_FILE = "models/v5_behavior_stats.json"

ALPHA = 1.0


def main():

    syscall_counts = Counter()
    transition_counts = Counter()

    total_events = 0
    total_transitions = 0
    invalid_events = 0

    # PID-aware previous syscall
    previous_by_pid = {}

    print()
    print("V5 BEHAVIOR STATISTICS")
    print("=" * 60)
    print("Input :", INPUT_FILE)
    print("Smoothing alpha :", ALPHA)
    print()

    with open(INPUT_FILE, "r") as f:

        for line in f:

            try:
                record = json.loads(line)

                syscall_id = int(
                    record["syscall_id"]
                )

                pid = int(
                    record["pid"]
                )

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError
            ):

                invalid_events += 1
                continue

            total_events += 1

            syscall_counts[
                syscall_id
            ] += 1

            if pid in previous_by_pid:

                previous = previous_by_pid[pid]

                transition = (
                    f"{previous},{syscall_id}"
                )

                transition_counts[
                    transition
                ] += 1

                total_transitions += 1

            previous_by_pid[pid] = syscall_id

    syscall_vocabulary = sorted(
        syscall_counts.keys()
    )

    transition_vocabulary = sorted(
        transition_counts.keys()
    )

    syscall_vocab_size = len(
        syscall_vocabulary
    )

    transition_vocab_size = len(
        transition_vocabulary
    )

    # --------------------------------------------------------
    # Smoothed syscall probabilities
    # --------------------------------------------------------

    syscall_denominator = (
        total_events
        + ALPHA * syscall_vocab_size
    )

    syscall_probabilities = {}

    for syscall_id in syscall_vocabulary:

        probability = (
            syscall_counts[syscall_id]
            + ALPHA
        ) / syscall_denominator

        syscall_probabilities[
            str(syscall_id)
        ] = probability

    # --------------------------------------------------------
    # Smoothed transition probabilities
    #
    # We smooth only over observed transition types.
    # Completely unseen transitions are handled separately
    # by the scorer using a conservative floor.
    # --------------------------------------------------------

    transition_denominator = (
        total_transitions
        + ALPHA * transition_vocab_size
    )

    transition_probabilities = {}

    for transition in transition_vocabulary:

        probability = (
            transition_counts[transition]
            + ALPHA
        ) / transition_denominator

        transition_probabilities[
            transition
        ] = probability

    output = {

        "source": INPUT_FILE,

        "total_events": total_events,

        "total_transitions": total_transitions,

        "invalid_events": invalid_events,

        "unique_syscalls": syscall_vocab_size,

        "unique_transitions": transition_vocab_size,

        "smoothing": {
            "method": "additive_laplace",
            "alpha": ALPHA
        },

        "syscall_counts": {
            str(k): int(v)
            for k, v in syscall_counts.items()
        },

        "syscall_probabilities": (
            syscall_probabilities
        ),

        "transition_counts": {
            k: int(v)
            for k, v in transition_counts.items()
        },

        "transition_probabilities": (
            transition_probabilities
        )
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

    print("Total events       :", total_events)
    print("Unique syscalls    :", syscall_vocab_size)
    print("Total transitions  :", total_transitions)
    print("Unique transitions :", transition_vocab_size)
    print("Invalid events     :", invalid_events)
    print()
    print(
        "Saved to           :",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
