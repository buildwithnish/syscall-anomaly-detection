import json
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


SCORES_FILE = "data/processed/anomaly_scores.npy"
THRESHOLD_FILE = "models/threshold.json"


def main():
    print("=" * 60)
    print("              DETECTOR EVALUATION")
    print("=" * 60)

    scores = np.load(SCORES_FILE)

    with open(THRESHOLD_FILE, "r") as file:
        threshold = json.load(file)["threshold"]

    # Since this dataset represents normal behavior,
    # every sequence has the ground-truth label NORMAL (0).
    actual = np.zeros(len(scores), dtype=int)

    predicted = (scores > threshold).astype(int)

    normal_count = np.sum(predicted == 0)
    anomaly_count = np.sum(predicted == 1)

    print(f"Total sequences : {len(scores)}")
    print(f"Threshold       : {threshold:.6f}")
    print()
    print(f"Predicted normal    : {normal_count}")
    print(f"Predicted anomalies : {anomaly_count}")

    print()
    print("Confusion Matrix:")
    print(confusion_matrix(actual, predicted))

    print()
    print("Classification Report:")

    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )
    )

    print("Metrics:")
    print(
        f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}"
    )

    print(
        f"Recall:    {recall_score(actual, predicted, zero_division=0):.4f}"
    )

    print(
        f"F1 Score:  {f1_score(actual, predicted, zero_division=0):.4f}"
    )


if __name__ == "__main__":
    main()
