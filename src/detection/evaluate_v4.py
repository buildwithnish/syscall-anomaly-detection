import json
import numpy as np


THRESHOLD_FILE = "models/v4_threshold.json"


def load_threshold():
    with open(THRESHOLD_FILE, "r") as f:
        data = json.load(f)

    return float(data["threshold"])


def evaluate_dataset(name, scores, threshold):
    detected = int(np.sum(scores > threshold))
    total = int(len(scores))
    missed = total - detected
    rate = detected / total if total else 0.0

    print()
    print(name)
    print("-" * 60)
    print("Total sequences :", total)
    print("Detected        :", detected)
    print("Missed          :", missed)
    print("Detection rate  :", f"{rate:.4%}")
    print("Mean score      :", f"{scores.mean():.6f}")
    print("Median score    :", f"{np.median(scores):.6f}")
    print("Max score       :", f"{scores.max():.6f}")


def main():

    threshold = load_threshold()

    normal_scores = np.load(
        "data/processed/v4_normal_test_scores.npy"
    )

    anomaly_1_scores = np.load(
        "data/processed/v4_anomaly_isolated_scores.npy"
    )

    anomaly_2_scores = np.load(
        "data/processed/v4_anomaly_second_scores.npy"
    )

    print()
    print("V4 FINAL EVALUATION")
    print("=" * 60)
    print("Frozen threshold :", f"{threshold:.6f}")

    # Normal test = false-positive evaluation
    false_positives = int(
        np.sum(normal_scores > threshold)
    )

    normal_total = int(
        len(normal_scores)
    )

    false_positive_rate = (
        false_positives / normal_total
        if normal_total
        else 0.0
    )

    print()
    print("HELD-OUT NORMAL TEST")
    print("-" * 60)
    print("Total sequences :", normal_total)
    print("False positives :", false_positives)
    print("FPR             :", f"{false_positive_rate:.4%}")

    evaluate_dataset(
        "ANOMALY WORKLOAD #1",
        anomaly_1_scores,
        threshold
    )

    evaluate_dataset(
        "ANOMALY WORKLOAD #2",
        anomaly_2_scores,
        threshold
    )


if __name__ == "__main__":
    main()
