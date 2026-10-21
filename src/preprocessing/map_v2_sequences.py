import json
import numpy as np

INPUT = "data/processed/v2_normal_sequences.npy"
CONFIG = "models/v2_model_config.json"
OUTPUT = "data/processed/v2_normal_sequences_mapped.npy"

sequences = np.load(INPUT)

with open(CONFIG, "r") as f:
    config = json.load(f)

known_syscalls = config["unique_syscalls"]
unk_index = config["unk_index"]

syscall_to_index = {
    syscall_id: index
    for index, syscall_id in enumerate(known_syscalls)
}

mapped = np.full(
    sequences.shape,
    unk_index,
    dtype=np.int64
)

for syscall_id, vocab_index in syscall_to_index.items():
    mapped[sequences == syscall_id] = vocab_index

unknown_events = int(
    (~np.isin(sequences, known_syscalls)).sum()
)

print("Original shape :", sequences.shape)
print("Known syscalls :", len(known_syscalls))
print("UNK index      :", unk_index)
print("Unknown events :", unknown_events)
print("Mapped min     :", mapped.min())
print("Mapped max     :", mapped.max())

np.save(OUTPUT, mapped)

print("Saved to        :", OUTPUT)
