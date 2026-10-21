import json
import os
import numpy as np

INPUT_FILE = "data/raw/v2_normal/normal_v2.jsonl"
OUTPUT_FILE = "data/processed/v2_normal_sequences.npy"

SEQUENCE_LENGTH = 20
STEP = 1

syscall_ids = []

with open(INPUT_FILE, "r") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)
        syscall_ids.append(record["syscall_id"])

print("Total syscall events:", len(syscall_ids))

if len(syscall_ids) < SEQUENCE_LENGTH:
    raise ValueError("Not enough syscall events to create sequences.")

sequences = []

for i in range(0, len(syscall_ids) - SEQUENCE_LENGTH + 1, STEP):
    sequences.append(
        syscall_ids[i:i + SEQUENCE_LENGTH]
    )

sequences = np.array(sequences, dtype=np.int64)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

np.save(OUTPUT_FILE, sequences)

print("Sequence length :", SEQUENCE_LENGTH)
print("Total sequences :", len(sequences))
print("Shape           :", sequences.shape)
print("Saved to        :", OUTPUT_FILE)
