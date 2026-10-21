import json
import os

import numpy as np


INPUT_FILE = "data/raw/anomaly_syscalls_sample.jsonl"
OUTPUT_FILE = "data/processed/anomaly_sequences.npy"

SEQUENCE_LENGTH = 20
STEP = 1


def main():
    print("=" * 50)
    print("       ANOMALY DATA PREPROCESSING")
    print("=" * 50)

    syscall_ids = []

    with open(INPUT_FILE, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            syscall_ids.append(
                int(record["syscall_id"])
            )

    print(f"Total events: {len(syscall_ids)}")

    sequences = []

    for start in range(
        0,
        len(syscall_ids) - SEQUENCE_LENGTH + 1,
        STEP
    ):
        sequence = syscall_ids[
            start:start + SEQUENCE_LENGTH
        ]

        sequences.append(sequence)

    sequences = np.array(
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

    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Shape: {sequences.shape}")
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
