import json
import os
from collections import Counter

import numpy as np


# ============================================================
# V4 BEHAVIOR STATISTICS
# ============================================================

INPUT_FILE = "data/raw/v2_normal/normal_v2.jsonl"

OUTPUT_FILE = "models/v4_behavior_stats.json"

# Ignore extremely rare events when calculating probabilities.
# They will still be represented in the statistics.
MIN_COUNT = 1


def main():

    print()
    print("V4 BEHAVIOR STATISTICS")
    print("=" * 60)

    syscall_counts = Counter()
    transition_counts = Counter()

    total_events = 0
    total_transitions = 0
    invalid_events = 0

    # --------------------------------------------------------
    # Read normal corpus
    # --------------------------------------------------------

    print("Reading normal corpus...")
    print("Input file:", INPUT_FILE)
    print()

    previous_syscall_by_pid = {}

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
                    invalid_events += 1
                    continue

                pid = int(pid)
                syscall_id = int(syscall_id)

                syscall_counts[syscall_id] += 1
                total_events += 1

                # ------------------------------------------------
                # PID-aware transition
                # ------------------------------------------------

                if pid in previous_syscall_by_pid:

                    previous_syscall = (
                        previous_syscall_by_pid[pid]
                    )

                    transition = (
                        previous_syscall,
                        syscall_id
                    )

                    transition_counts[transition] += 1
                    total_transitions += 1

                previous_syscall_by_pid[pid] = syscall_id

            except (
                json.JSONDecodeError,
                ValueError,
                TypeError
            ):
                invalid_events += 1

    # --------------------------------------------------------
    # Calculate probabilities
    # --------------------------------------------------------

    syscall_probabilities = {}

    for syscall_id, count in syscall_counts.items():

        if count >= MIN_COUNT:

            syscall_probabilities[str(syscall_id)] = (
                count / total_events
            )

    transition_probabilities = {}

    for transition, count in transition_counts.items():

        if count >= MIN_COUNT:

            previous_syscall, current_syscall = transition

            key = f"{previous_syscall},{current_syscall}"

            transition_probabilities[key] = (
                count / total_transitions
            )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    stats = {
        "source": INPUT_FILE,

        "total_events": total_events,

        "total_transitions": total_transitions,

        "invalid_events": invalid_events,

        "unique_syscalls": len(syscall_counts),

        "unique_transitions": len(transition_counts),

        "syscall_counts": {
            str(k): int(v)
            for k, v in syscall_counts.items()
        },

        "syscall_probabilities": syscall_probabilities,

        "transition_counts": {
            f"{k[0]},{k[1]}": int(v)
            for k, v in transition_counts.items()
        },

        "transition_probabilities": transition_probabilities
    }

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as f:

        json.dump(
            stats,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("RESULT")
    print("=" * 60)

    print(
        "Total events       :",
        total_events
    )

    print(
        "Invalid events     :",
        invalid_events
    )

    print(
        "Unique syscalls    :",
        len(syscall_counts)
    )

    print(
        "Total transitions  :",
        total_transitions
    )

    print(
        "Unique transitions :",
        len(transition_counts)
    )

    print()
    print(
        "Statistics saved to:",
        OUTPUT_FILE
    )

    print()
    print("V4 behavior statistics complete.")


if __name__ == "__main__":
    main()
