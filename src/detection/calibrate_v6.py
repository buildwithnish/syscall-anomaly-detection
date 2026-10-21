import numpy as np
import json

CALIBRATION_FILE = "data/processed/normal_calibration_sequences.npy"

V5_NORMALIZATION = "models/v5_normalization.json"

OUTPUT_FILE = "models/v6_calibration.json"

scores = np.load(
    "data/processed/v5_normal_calibration_scores.npy"
)

print("V6 CALIBRATION")
print("=" * 60)
print(f"Calibration sequences : {len(scores)}")

# Use the V5 hybrid score only as the input feature for the
# first calibrated V6 baseline. This keeps the experiment
# reproducible while separating calibration from thresholding.

threshold = float(np.percentile(scores, 99))

result = {
    "method": "normal_calibration_99th_percentile",
    "feature": "v5_hybrid_score",
    "threshold": threshold,
    "target_fpr": 0.01,
    "calibration_sequences": int(len(scores))
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(result, f, indent=4)

print(f"Threshold             : {threshold:.6f}")
print(f"Target FPR            : 1.00%")
print(f"Saved to              : {OUTPUT_FILE}")
