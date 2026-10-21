import json
import numpy as np


NORMAL_SCORES = "data/processed/v4_normal_test_scores.npy"
CALIBRATION_SCORES = "data/processed/v4_normal_calibration_scores.npy"
ANOMALY_1_SCORES = "data/processed/v4_anomaly_isolated_scores.npy"
ANOMALY_2_SCORES = "data/processed/v4_anomaly_second_scores.npy"

THRESHOLD_FILE = "models/v4_threshold.json"


def describe(name, scores, threshold):
    print()
    print(name)
    print("-" * 60)

    print("Count       :", len(scores))
    print("Min         :", f"{scores.min():.6f}")
    print("Mean        :", f"{scores.mean():.6f}")
    print("Median      :", f"{np.median(scores):.6f}")
    print("95th pct    :", f"{np.percentile(scores, 95):.6f}")
    print("99th pct    :", f"{np.percentile(scores, 99):.6f}")
    print("99.5th pct  :", f"{np.percentile(scores, 99.5):.6f}")
    print("Max         :", f"{scores.max():.6f}")

    detected = int(np.sum(scores > threshold))

    print("Above thresh:", detected)
    print(
        "Rate        :",
        f"{detected / len(scores):.4%}"
    )


def main():

    with open(THRESHOLD_FILE, "r") as f:
        threshold = float(
            json.load(f)["threshold"]
        )

    calibration = np.load(
        CALIBRATION_SCORES
    )

    normal = np.load(
        NORMAL_SCORES
    )

    anomaly_1 = np.load(
        ANOMALY_1_SCORES
    )

    anomaly_2 = np.load(
        ANOMALY_2_SCORES
    )

    print()
    print("V4 SCORE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(
        "Frozen threshold:",
        f"{threshold:.6f}"
    )

    describe(
        "NORMAL CALIBRATION",
        calibration,
        threshold
    )

    describe(
        "HELD-OUT NORMAL",
        normal,
        threshold
    )

    describe(
        "ANOMALY WORKLOAD #1",
        anomaly_1,
        threshold
    )

    describe(
        "ANOMALY WORKLOAD #2",
        anomaly_2,
        threshold
    )


if __name__ == "__main__":
    main()
