import json
import os
import sys
from collections import defaultdict

import numpy as np


# ============================================================
# V3 PID-AWARE SEQUENCE PREPROCESSING
# ============================================================

INPUT_FILE = "data/raw/v2_normal/normal_v2.jsonl"

OUTPUT_FILE = "data/processed/v3_pid_sequences.npy"

SEQUENCE_LENGTH = 20
STEP = 1

# Minimum number of events required for a PID to produce sequences
MIN_EVENTS_PER_PID = SEQUENCE_LENGTH


def load_events_by_pid(input_file):
    """
    Read syscall events and group syscall IDs by PID.

    Each PID gets its own independent chronological syscall stream.
    This prevents sequences from mixing unrelated processes.
    """

    events_by_pid = defaultdict(list)

    total_events = 0
    invalid_events = 0

    print("Loading syscall events...")
    print("Input file:", input_file)
    print()

    with open(input_file, "r") as f:
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

                events_by_pid[int(pid)].append(int(syscall_id))
                total_events += 1

            except (json.JSONDecodeError, ValueError, TypeError):
                invalid_events += 1

    return events_by_pid, total_events, invalid_events


def create_pid_sequences(events_by_pid):
    """
    Create fixed-length sequences independently for each PID.

    No sequence is allowed to cross a PID boundary.
    """

    sequences = []

    eligible_pids = 0
    skipped_pids = 0

    for pid, syscall_ids in events_by_pid.items():

        if len(syscall_ids) < MIN_EVENTS_PER_PID:
            skipped_pids += 1
            continue

        eligible_pids += 1

        for start in range(
            0,
            len(syscall_ids) - SEQUENCE_LENGTH + 1,
            STEP
        ):
            sequence = syscall_ids[
                start:start + SEQUENCE_LENGTH
            ]

            sequences.append(sequence)

    return (
        sequences,
        eligible_pids,
        skipped_pids
    )


def main():

    print()
    print("V3 PID-AWARE PREPROCESSING")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print("ERROR: Input file not found:")
        print(INPUT_FILE)
        sys.exit(1)

    events_by_pid, total_events, invalid_events = (
        load_events_by_pid(INPUT_FILE)
    )

    print("Total valid events :", total_events)
    print("Invalid events     :", invalid_events)
    print("Unique PIDs        :", len(events_by_pid))

    print()
    print("Creating PID-isolated sequences...")

    sequences, eligible_pids, skipped_pids = (
        create_pid_sequences(events_by_pid)
    )

    print("Eligible PIDs      :", eligible_pids)
    print("Skipped PIDs       :", skipped_pids)

    if not sequences:
        print()
        print("ERROR: No sequences were created.")
        sys.exit(1)

    sequences = np.asarray(
        sequences,
        dtype=np.int64
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    np.save(
        OUTPUT_FILE,
        sequences
    )

    print()
    print("RESULT")
    print("=" * 60)
    print("Sequence length    :", SEQUENCE_LENGTH)
    print("Total sequences    :", len(sequences))
    print("Shape              :", sequences.shape)
    print("Data type          :", sequences.dtype)
    print()
    print("Saved to:")
    print(OUTPUT_FILE)
    print()
    print("V3 PID-aware preprocessing complete.")


if __name__ == "__main__":
    main()
