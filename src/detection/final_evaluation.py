import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

NORMAL_SCORES = "data/processed/normal_test_scores.npy"
FIRST_ANOMALY_SCORES = "data/processed/anomaly_isolated_scores.npy"
SECOND_ANOMALY_SCORES = "data/processed/anomaly_second_scores.npy"
THRESHOLD_FILE = "models/threshold.json"

OUTPUT_DIR = Path("evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Load scores
# ============================================================

normal_scores = np.load(NORMAL_SCORES)
first_anomaly_scores = np.load(FIRST_ANOMALY_SCORES)
second_anomaly_scores = np.load(SECOND_ANOMALY_SCORES)

with open(THRESHOLD_FILE, "r") as file:
    threshold_config = json.load(file)

threshold = float(threshold_config["threshold"])


# ============================================================
# Calculate metrics
# ============================================================

normal_detected = int((normal_scores >= threshold).sum())
normal_fpr = normal_detected / len(normal_scores)

first_detected = int(
    (first_anomaly_scores >= threshold).sum()
)

first_detection_rate = (
    first_detected / len(first_anomaly_scores)
)

second_detected = int(
    (second_anomaly_scores >= threshold).sum()
)

second_detection_rate = (
    second_detected / len(second_anomaly_scores)
)


metrics = {
    "model": {
        "architecture": "LSTM Autoencoder",
        "embedding_dim": 32,
        "hidden_dim": 64,
        "sequence_length": 20,
        "vocabulary_size": 110,
        "unknown_token": "<UNK>"
    },
    "threshold": {
        "value": threshold,
        "method": "normal_calibration_99th_percentile",
        "target_fpr": 0.01
    },
    "normal_test": {
        "sequences": len(normal_scores),
        "false_positives": normal_detected,
        "false_positive_rate": normal_fpr
    },
    "isolated_workload_1": {
        "sequences": len(first_anomaly_scores),
        "detected": first_detected,
        "missed": len(first_anomaly_scores) - first_detected,
        "detection_rate": first_detection_rate
    },
    "isolated_workload_2": {
        "sequences": len(second_anomaly_scores),
        "detected": second_detected,
        "missed": len(second_anomaly_scores) - second_detected,
        "detection_rate": second_detection_rate
    }
}


with open(OUTPUT_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# ============================================================
# Console summary
# ============================================================

print("=" * 70)
print("FINAL SYSCALL ANOMALY DETECTION EVALUATION")
print("=" * 70)

print()
print("MODEL")
print("-" * 70)
print("Architecture       : LSTM Autoencoder")
print("Embedding dimension: 32")
print("Hidden dimension   : 64")
print("Sequence length    : 20")
print("Vocabulary size    : 110")

print()
print("THRESHOLD")
print("-" * 70)
print(f"Threshold          : {threshold:.6f}")
print("Calibration        : Normal-only 99th percentile")

print()
print("HELD-OUT NORMAL TEST")
print("-" * 70)
print(f"Sequences          : {len(normal_scores)}")
print(f"False positives    : {normal_detected}")
print(f"False positive rate: {normal_fpr:.4%}")

print()
print("ISOLATED WORKLOAD #1")
print("-" * 70)
print(f"Sequences          : {len(first_anomaly_scores)}")
print(f"Detected           : {first_detected}")
print(f"Missed             : {len(first_anomaly_scores) - first_detected}")
print(f"Detection rate     : {first_detection_rate:.4%}")

print()
print("ISOLATED WORKLOAD #2")
print("-" * 70)
print(f"Sequences          : {len(second_anomaly_scores)}")
print(f"Detected           : {second_detected}")
print(f"Missed             : {len(second_anomaly_scores) - second_detected}")
print(f"Detection rate     : {second_detection_rate:.4%}")


# ============================================================
# Plot 1 — Detection rate comparison
# ============================================================

labels = [
    "Normal\n(FPR)",
    "Workload 1",
    "Workload 2"
]

values = [
    normal_fpr * 100,
    first_detection_rate * 100,
    second_detection_rate * 100
]

plt.figure(figsize=(9, 6))
plt.bar(labels, values)
plt.ylabel("Rate (%)")
plt.title("Final Detection Evaluation")
plt.ylim(0, 105)

for i, value in enumerate(values):
    plt.text(
        i,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "detection_rate_comparison.png",
    dpi=200
)
plt.close()


# ============================================================
# Plot 2 — Score distributions
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    normal_scores,
    bins=100,
    alpha=0.6,
    label="Normal test"
)

plt.hist(
    first_anomaly_scores,
    bins=100,
    alpha=0.6,
    label="Isolated workload 1"
)

plt.hist(
    second_anomaly_scores,
    bins=100,
    alpha=0.6,
    label="Isolated workload 2"
)

plt.axvline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

plt.xlabel("Reconstruction error")
plt.ylabel("Number of sequences")
plt.title("Anomaly Score Distributions")
plt.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "score_distributions.png",
    dpi=200
)
plt.close()


# ============================================================
# Plot 3 — Score distributions on logarithmic x-axis
# ============================================================

plt.figure(figsize=(10, 6))

positive_normal = normal_scores[
    normal_scores > 0
]

positive_first = first_anomaly_scores[
    first_anomaly_scores > 0
]

positive_second = second_anomaly_scores[
    second_anomaly_scores > 0
]

plt.hist(
    positive_normal,
    bins=100,
    alpha=0.6,
    label="Normal test"
)

plt.hist(
    positive_first,
    bins=100,
    alpha=0.6,
    label="Isolated workload 1"
)

plt.hist(
    positive_second,
    bins=100,
    alpha=0.6,
    label="Isolated workload 2"
)

plt.axvline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

plt.xscale("log")
plt.xlabel("Reconstruction error (log scale)")
plt.ylabel("Number of sequences")
plt.title("Anomaly Score Distributions — Log Scale")
plt.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "score_distributions_log.png",
    dpi=200
)
plt.close()


# ============================================================
# Plot 4 — Percentile comparison
# ============================================================

percentiles = [
    50,
    75,
    90,
    95,
    97,
    98,
    99,
    99.5,
    99.9
]

normal_percentiles = np.percentile(
    normal_scores,
    percentiles
)

first_percentiles = np.percentile(
    first_anomaly_scores,
    percentiles
)

second_percentiles = np.percentile(
    second_anomaly_scores,
    percentiles
)

plt.figure(figsize=(10, 6))

plt.plot(
    percentiles,
    normal_percentiles,
    marker="o",
    label="Normal test"
)

plt.plot(
    percentiles,
    first_percentiles,
    marker="o",
    label="Isolated workload 1"
)

plt.plot(
    percentiles,
    second_percentiles,
    marker="o",
    label="Isolated workload 2"
)

plt.axhline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

plt.xlabel("Percentile")
plt.ylabel("Reconstruction error")
plt.title("Score Percentile Comparison")
plt.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "percentile_comparison.png",
    dpi=200
)
plt.close()


# ============================================================
# README metrics
# ============================================================

readme_section = f"""
## Evaluation

The model was evaluated using a normal-only calibration set, a
held-out normal test set, and two independently collected
PID-isolated controlled workloads.

### Configuration

- Model: LSTM Autoencoder
- Embedding dimension: 32
- Hidden dimension: 64
- Sequence length: 20
- Vocabulary: 109 known syscalls + `<UNK>`
- Threshold: `{threshold:.6f}`
- Threshold method: normal-only 99th percentile calibration

### Results

| Dataset | Sequences | Result |
|---|---:|---:|
| Held-out normal | {len(normal_scores):,} | {normal_fpr:.4%} false-positive rate |
| Isolated workload 1 | {len(first_anomaly_scores):,} | {first_detection_rate:.4%} detected |
| Independent workload 2 | {len(second_anomaly_scores):,} | {second_detection_rate:.4%} detected |

### Interpretation

The detector successfully identifies highly out-of-distribution
syscall sequences in the first controlled isolated workload.
The independent second workload shows substantially lower
sequence-level detection, demonstrating that anomaly sensitivity
depends on the syscall distribution of the workload.

The threshold and model were kept frozen during these experiments.
The controlled workloads should not be interpreted as confirmed
malicious attacks.

### Limitations

- Evaluation workloads are controlled experiments rather than
  real-world attack traces.
- Sequence-level detection does not directly represent
  process-level attack detection.
- The held-out normal false-positive rate was higher than the
  1% calibration target.
- Additional datasets containing real benign applications and
  representative attack traces are required for production
  security evaluation.
"""

with open(OUTPUT_DIR / "README_EVALUATION.md", "w") as file:
    file.write(readme_section)


# ============================================================
# Complete
# ============================================================

print()
print("=" * 70)
print("FILES GENERATED")
print("=" * 70)
print(f"Metrics JSON       : {OUTPUT_DIR / 'metrics.json'}")
print(
    f"Detection plot     : "
    f"{OUTPUT_DIR / 'detection_rate_comparison.png'}"
)
print(
    f"Score distribution : "
    f"{OUTPUT_DIR / 'score_distributions.png'}"
)
print(
    f"Log distribution   : "
    f"{OUTPUT_DIR / 'score_distributions_log.png'}"
)
print(
    f"Percentile plot    : "
    f"{OUTPUT_DIR / 'percentile_comparison.png'}"
)
print(
    f"README section     : "
    f"{OUTPUT_DIR / 'README_EVALUATION.md'}"
)

print("=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)
