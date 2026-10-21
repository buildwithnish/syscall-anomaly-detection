import json
import os
import numpy as np


INPUT_FILE = "data/raw/anomaly_syscalls_second.jsonl"
OUTPUT_FILE = "data/processed/anomaly_second_sequences.npy"

SEQUENCE_LENGTH = 20
STEP = 1


def load_syscalls():
    syscall_ids = []

    with open(INPUT_FILE, "r") as file:
        for line in file:
            record = json.loads(line)
            syscall_ids.append(record["syscall_id"])

    return syscall_ids


def create_sequences(syscalls):
    sequences = []

    for i in range(
        0,
        len(syscalls) - SEQUENCE_LENGTH + 1,
        STEP
    ):
        sequence = syscalls[i:i + SEQUENCE_LENGTH]
        sequences.append(sequence)

    return np.array(sequences, dtype=np.int64)


syscalls = load_syscalls()

print("=" * 60)
print("   ISOLATED ANOMALY DATA PREPROCESSING")
print("=" * 60)
print(f"Total events     : {len(syscalls)}")
print(f"Sequence length  : {SEQUENCE_LENGTH}")

sequences = create_sequences(syscalls)

print(f"Total sequences  : {len(sequences)}")
print(f"Shape            : {sequences.shape}")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

np.save(OUTPUT_FILE, sequences)

print()
print(f"Saved to         : {OUTPUT_FILE}")
print("=" * 60)
