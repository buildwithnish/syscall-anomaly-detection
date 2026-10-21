import json
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_calibration_scores.npy"
)

THRESHOLD_FILE = (
    PROJECT_ROOT
    / "models"
    / "threshold.json"
)


# ============================================================
# LOAD SCORES
# ============================================================

scores = np.load(SCORES_FILE)


# ============================================================
# CANDIDATE PERCENTILES
# ============================================================

percentiles = [
    95,
    97,
    98,
    99,
    99.5,
    99.9
]


print("=" * 70)
print("NORMAL-ONLY THRESHOLD CALIBRATION")
print("=" * 70)

print(
    f"Calibration sequences : {len(scores)}"
)

print()
print(
    f"{'Percentile':<12}"
    f"{'Threshold':<15}"
    f"{'Expected FPR':<15}"
    f"{'Expected alerts':<15}"
)

print("-" * 57)


candidates = []

for percentile in percentiles:

    threshold = float(
        np.percentile(
            scores,
            percentile
        )
    )

    expected_fpr = (
        100.0 - percentile
    )

    expected_alerts = int(
        np.sum(scores >= threshold)
    )

    candidates.append(
        {
            "percentile": percentile,
            "threshold": threshold,
            "expected_fpr": expected_fpr,
            "expected_alerts": expected_alerts
        }
    )

    print(
        f"{percentile:<12}"
        f"{threshold:<15.6f}"
        f"{expected_fpr:<15.2f}%"
        f"{expected_alerts:<15}"
    )


# ============================================================
# RECOMMENDED OPERATING POINT
# ============================================================

# Target approximately 1% alerts on normal calibration data.
recommended = min(
    candidates,
    key=lambda x: abs(
        x["expected_fpr"] - 1.0
    )
)


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
    f"Calibration FPR  : "
    f"{recommended['expected_fpr']:.2f}%"
)

print(
    f"Expected alerts  : "
    f"{recommended['expected_alerts']}"
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_config = {

    "threshold": recommended["threshold"],

    "method":
        "normal_calibration_99th_percentile",

    "calibration_dataset":
        "normal_calibration_sequences.npy",

    "calibration_sequences":
        int(len(scores)),

    "target_fpr":
        0.01,

    "observed_calibration_fpr":
        recommended["expected_fpr"] / 100.0
}


with open(
    THRESHOLD_FILE,
    "w"
) as f:

    json.dump(
        threshold_config,
        f,
        indent=4
    )


print()
print(
    f"Threshold saved : {THRESHOLD_FILE}"
)

print("=" * 70)
print("CALIBRATION COMPLETE")
print("=" * 70)
