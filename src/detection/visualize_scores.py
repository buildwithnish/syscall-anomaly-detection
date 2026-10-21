import json

import numpy as np
import matplotlib.pyplot as plt


SCORES_FILE = "data/processed/anomaly_scores.npy"
THRESHOLD_FILE = "models/threshold.json"
OUTPUT_FILE = "data/processed/anomaly_score_distribution.png"


scores = np.load(SCORES_FILE)

with open(THRESHOLD_FILE, "r") as file:
    threshold = json.load(file)["threshold"]


plt.figure(figsize=(10, 6))

plt.hist(
    scores,
    bins=50
)

plt.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold:.4f}"
)

plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Sequences")
plt.title("Syscall Anomaly Score Distribution")
plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=150)

plt.show()

print(f"Chart saved to: {OUTPUT_FILE}")
