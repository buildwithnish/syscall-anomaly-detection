import os
import json
import numpy as np


# ============================================================
# Configuration
# ============================================================

NORMAL_SCORES_FILE = (
    "data/processed/normal_test_scores.npy"
)

ANOMALY_SCORES_FILE = (
    "data/processed/anomaly_dataset_scores.npy"
)

THRESHOLD_FILE = "models/threshold.json"


# ============================================================
# Load scores
# ============================================================

if not os.path.exists(NORMAL_SCORES_FILE):
    raise FileNotFoundError(
        f"Normal scores not found: {NORMAL_SCORES_FILE}"
    )

if not os.path.exists(ANOMALY_SCORES_FILE):
    raise FileNotFoundError(
        f"Anomaly scores not found: {ANOMALY_SCORES_FILE}"
    )


normal_scores = np.load(
    NORMAL_SCORES_FILE
)

anomaly_scores = np.load(
    ANOMALY_SCORES_FILE
)


# ============================================================
# Calibration
# ============================================================

print("=" * 70)
print("THRESHOLD CALIBRATION")
print("=" * 70)

print(
    f"Normal sequences  : {len(normal_scores)}"
)

print(
    f"Anomaly sequences : {len(anomaly_scores)}"
)

print()


# ============================================================
# Candidate operating points
# ============================================================

percentiles = [
    95,
    97,
    98,
    99,
    99.5,
    99.9
]


results = []


print(
    f"{'Percentile':<12}"
    f"{'Threshold':<15}"
    f"{'Normal FPR':<15}"
    f"{'Anomaly Recall':<18}"
    f"{'Detected':<12}"
)

print("-" * 70)


for percentile in percentiles:

    threshold = float(
        np.percentile(
            normal_scores,
            percentile
        )
    )

    # --------------------------------------------------------
    # Normal false positives
    # --------------------------------------------------------

    normal_false_positives = int(
        np.sum(
            normal_scores > threshold
        )
    )

    normal_fpr = (
        normal_false_positives
        / len(normal_scores)
    )

    # --------------------------------------------------------
    # Anomaly detection rate
    # --------------------------------------------------------

    detected_anomalies = int(
        np.sum(
            anomaly_scores > threshold
        )
    )

    anomaly_recall = (
        detected_anomalies
        / len(anomaly_scores)
    )

    print(
        f"{percentile:<12.1f}"
        f"{threshold:<15.6f}"
        f"{normal_fpr * 100:<15.2f}%"
        f"{anomaly_recall * 100:<18.2f}%"
        f"{detected_anomalies:<12}"
    )

    results.append(
        {
            "percentile": percentile,
            "threshold": threshold,
            "normal_fpr": normal_fpr,
            "normal_false_positives":
                normal_false_positives,
            "anomaly_recall":
                anomaly_recall,
            "detected_anomalies":
                detected_anomalies
        }
    )


# ============================================================
# Select recommended threshold
# ============================================================
#
# We want approximately 1% false positives on unseen
# normal traffic.
#
# ============================================================

recommended = min(
    results,
    key=lambda x: abs(
        x["normal_fpr"] - 0.01
    )
)


# ============================================================
# Save recommended threshold
# ============================================================

threshold_data = {

    "threshold": recommended["threshold"],

    "method":
        "normal_test_99th_percentile",

    "target_false_positive_rate":
        0.01,

    "observed_false_positive_rate":
        recommended["normal_fpr"],

    "normal_false_positives":
        recommended["normal_false_positives"],

    "normal_sequences":
        len(normal_scores),

    "anomaly_sequences":
        len(anomaly_scores),

    "anomaly_detection_rate":
        recommended["anomaly_recall"],

    "detected_anomalies":
        recommended["detected_anomalies"],

    "candidate_thresholds":
        results
}


with open(
    THRESHOLD_FILE,
    "w"
) as file:

    json.dump(
        threshold_data,
        file,
        indent=4
    )


# ============================================================
# Final recommendation
# ============================================================

print()
print("=" * 70)
print("RECOMMENDED OPERATING POINT")
print("=" * 70)

print(
    f"Target FPR       : 1.00%"
)

print(
    f"Threshold        : "
    f"{recommended['threshold']:.6f}"
)

print(
    f"Observed FPR     : "
    f"{recommended['normal_fpr'] * 100:.2f}%"
)

print(
    f"Anomaly recall   : "
    f"{recommended['anomaly_recall'] * 100:.2f}%"
)

print(
    f"Anomalies caught : "
    f"{recommended['detected_anomalies']}"
)

print(
    f"Total anomalies  : "
    f"{len(anomaly_scores)}"
)

print()
print(
    f"Threshold saved  : {THRESHOLD_FILE}"
)

print("=" * 70)
print("CALIBRATION COMPLETE")
print("=" * 70)
