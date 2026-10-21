import json
import os
import numpy as np


INPUT_FILE = "data/raw/normal_calibration_syscalls.jsonl"
OUTPUT_FILE = "data/processed/normal_calibration_sequences.npy"

SEQUENCE_LENGTH = 20
STEP = 1


def load_syscalls():

    syscalls = []

    with open(INPUT_FILE, "r") as f:

        for line in f:

            record = json.loads(line)

            syscalls.append(
                int(record["syscall_id"])
            )

    return syscalls


def create_sequences(syscalls):

    sequences = []

    for i in range(
        0,
        len(syscalls) - SEQUENCE_LENGTH + 1,
        STEP
    ):

        sequence = syscalls[
            i:i + SEQUENCE_LENGTH
        ]

        sequences.append(sequence)

    return np.array(
        sequences,
        dtype=np.int64
    )


def main():

    print("=" * 60)
    print("CALIBRATION DATA PREPROCESSING")
    print("=" * 60)

    syscalls = load_syscalls()

    print(
        f"Loaded syscall events : {len(syscalls)}"
    )

    sequences = create_sequences(syscalls)

    print(
        f"Generated sequences   : {len(sequences)}"
    )

    print(
        f"Sequence shape         : {sequences.shape}"
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    np.save(
        OUTPUT_FILE,
        sequences
    )

    print(
        f"Saved                 : {OUTPUT_FILE}"
    )

    print("=" * 60)
    print("CALIBRATION PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
