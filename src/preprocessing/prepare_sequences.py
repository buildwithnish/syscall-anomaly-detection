# ============================================================
# SYSCALL SEQUENCE PREPROCESSING
# ============================================================
#
# Input:
#     data/raw/syscalls.jsonl
#
# Output:
#     data/processed/sequences.npy
#     data/processed/metadata.json
#
# The raw syscall stream:
#
#     openat -> read -> close -> write -> ...
#
# is converted into fixed-length numerical sequences:
#
#     [257, 0, 3, 1, ...]
#
# These sequences will later be used to train the
# LSTM Autoencoder.
#
# ============================================================


import json
import os

import numpy as np


# ============================================================
# PATHS
# ============================================================
INPUT_FILE = "data/raw/syscalls_sample.jsonl"

OUTPUT_DIR = "data/processed"

SEQUENCES_FILE = os.path.join(
    OUTPUT_DIR,
    "sequences.npy"
)

METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)


# ============================================================
# SEQUENCE SETTINGS
# ============================================================

# Number of syscalls in each sequence
SEQUENCE_LENGTH = 20

# Step between consecutive sequences
#
# Example:
#
# Sequence 1:
# syscall 0 -> 19
#
# Sequence 2:
# syscall 1 -> 20
#
# Sequence 3:
# syscall 2 -> 21
#
# This creates overlapping windows.
#
STEP = 1


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input dataset not found: {INPUT_FILE}\n"
        "Run the dataset collector first."
    )


# ============================================================
# LOAD SYSCALL DATA
# ============================================================

print("==============================================")
print("       SYSCALL DATA PREPROCESSING")
print("==============================================")
print()

print(f"Reading dataset: {INPUT_FILE}")
print()


syscall_ids = []

event_count = 0


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        try:

            record = json.loads(line)

        except json.JSONDecodeError:

            print(
                "Warning: skipping invalid JSON line."
            )

            continue


        # Get syscall ID
        syscall_id = record.get(
            "syscall_id"
        )


        # Make sure syscall ID exists
        if syscall_id is None:

            continue


        # Convert to integer
        syscall_id = int(
            syscall_id
        )


        # Add to list
        syscall_ids.append(
            syscall_id
        )

        event_count += 1


# ============================================================
# CHECK DATA
# ============================================================

print(
    f"Total valid syscall events: {event_count}"
)

print()


if len(syscall_ids) < SEQUENCE_LENGTH:

    raise ValueError(
        f"Not enough syscall events.\n"
        f"Need at least {SEQUENCE_LENGTH}, "
        f"but only found {len(syscall_ids)}."
    )


# ============================================================
# CONVERT TO NUMPY ARRAY
# ============================================================

syscall_array = np.array(
    syscall_ids,
    dtype=np.int32
)


# ============================================================
# CREATE SLIDING WINDOWS
# ============================================================

print(
    f"Creating sequences of length: "
    f"{SEQUENCE_LENGTH}"
)

print(
    f"Sequence step: {STEP}"
)

print()


sequences = []


for start in range(
    0,
    len(syscall_array) - SEQUENCE_LENGTH + 1,
    STEP
):

    end = start + SEQUENCE_LENGTH

    sequence = syscall_array[
        start:end
    ]

    sequences.append(
        sequence
    )


# ============================================================
# CONVERT SEQUENCES TO NUMPY ARRAY
# ============================================================

sequences = np.array(
    sequences,
    dtype=np.int32
)


# ============================================================
# SAVE SEQUENCES
# ============================================================

np.save(
    SEQUENCES_FILE,
    sequences
)


# ============================================================
# CREATE METADATA
# ============================================================

unique_syscalls = np.unique(
    syscall_array
)


metadata = {

    "input_file": INPUT_FILE,

    "sequence_length": SEQUENCE_LENGTH,

    "step": STEP,

    "total_events": int(
        len(syscall_array)
    ),

    "total_sequences": int(
        len(sequences)
    ),

    "unique_syscalls": int(
        len(unique_syscalls)
    ),

    "min_syscall_id": int(
        np.min(syscall_array)
    ),

    "max_syscall_id": int(
        np.max(syscall_array)
    )
}


# ============================================================
# SAVE METADATA
# ============================================================

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("==============================================")
print("Preprocessing completed successfully!")
print("==============================================")
print()

print(
    f"Total syscall events : {len(syscall_array)}"
)

print(
    f"Unique syscalls      : "
    f"{len(unique_syscalls)}"
)

print(
    f"Sequence length      : "
    f"{SEQUENCE_LENGTH}"
)

print(
    f"Total sequences      : "
    f"{len(sequences)}"
)

print()

print(
    f"Sequences saved to:"
)

print(
    f"  {SEQUENCES_FILE}"
)

print()

print(
    f"Metadata saved to:"
)

print(
    f"  {METADATA_FILE}"
)

print()


# ============================================================
# SHOW FIRST SEQUENCE
# ============================================================

print("First sequence:")

print(
    sequences[0]
)

print()

print("First sequence shape:")

print(
    sequences[0].shape
)

print()

print("Complete dataset shape:")

print(
    sequences.shape
)

print()
print("Done!")
