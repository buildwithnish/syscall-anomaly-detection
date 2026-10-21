import json
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


NORMAL_SCORES_FILE = "data/processed/anomaly_scores.npy"
ANOMALY_SCORES_FILE = "data/processed/anomaly_dataset_scores.npy"
THRESHOLD_FILE = "models/threshold.json"


def main():

    print("=" * 60)
    print("        FINAL ANOMALY DETECTOR EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # Load scores
    # --------------------------------------------------

    normal_scores = np.load(
        NORMAL_SCORES_FILE
    )

    anomaly_scores = np.load(
        ANOMALY_SCORES_FILE
    )

    # --------------------------------------------------
    # Load threshold
    # --------------------------------------------------

    with open(
        THRESHOLD_FILE,
        "r"
    ) as file:

        threshold_config = json.load(file)

    threshold = threshold_config["threshold"]

    print()
    print(f"Normal sequences  : {len(normal_scores)}")
    print(f"Anomaly sequences : {len(anomaly_scores)}")
    print(f"Threshold         : {threshold:.6f}")

    # --------------------------------------------------
    # Create labels
    # --------------------------------------------------

    y_true = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(anomaly_scores))
    ])

    scores = np.concatenate([
        normal_scores,
        anomaly_scores
    ])

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = (
        scores > threshold
    ).astype(int)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        scores
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print()
    print("                 Predicted")
    print("                 Normal   Anomaly")
    print(
        f"Actual Normal    {cm[0][0]:6d}   {cm[0][1]:7d}"
    )
    print(
        f"Actual Anomaly   {cm[1][0]:6d}   {cm[1][1]:7d}"
    )

    print()
    print("=" * 60)
    print("METRICS")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    # --------------------------------------------------
    # Detection rate
    # --------------------------------------------------

    detected_anomalies = np.sum(
        anomaly_scores > threshold
    )

    detection_rate = (
        detected_anomalies /
        len(anomaly_scores)
    )

    print()
    print(f"Anomaly detection rate : {detection_rate:.4f}")
    print(
        f"Anomalies detected     : "
        f"{detected_anomalies}/{len(anomaly_scores)}"
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print()

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Anomaly"
            ],
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
