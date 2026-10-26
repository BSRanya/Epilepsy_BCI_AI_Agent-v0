"""
Train a Random Forest on CHB-MIT features (Multi-File / Cross-Session Split).
Trains on one file, tests on a completely different file from the same patient.
"""
import json
import os
import platform
from datetime import datetime, timezone

import numpy as np
import mne
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from src.tools.features import _extract_features

# ------------------------------------------------------------------
# 1. Config
# ------------------------------------------------------------------
TRAIN_EDF     = "data/data/chb01_03.edf"
TEST_EDF      = "data/data/chb01_04.edf"
SUMMARY_PATH  = "data/data/chb01-summary.txt"
MODEL_PATH    = "src/models/seizure_rf_model.pkl"
META_PATH     = "src/models/seizure_rf_model.meta.json"

TARGET_CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "FP2-F4"]
WIN_SEC         = 5
STRIDE_SEC      = 1


# ------------------------------------------------------------------
# 2. Parser
# ------------------------------------------------------------------
def parse_seizures(summary_path: str, edf_filename: str) -> list[tuple[int, int]]:
    seizures = []
    current_file = None
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("File Name:"):
                current_file = line.replace("File Name:", "").strip()
                continue
            if current_file == edf_filename:
                if "Start Time:" in line and "Seizure" in line:
                    start_str = line.split("Start Time:")[1].replace("seconds", "").strip()
                    seizures.append([int(start_str), None])
                elif "End Time:" in line and "Seizure" in line and seizures:
                    end_str = line.split("End Time:")[1].replace("seconds", "").strip()
                    seizures[-1][1] = int(end_str)
    return [(s, e) for s, e in seizures if e is not None]


# ------------------------------------------------------------------
# 3. Reusable File Processor
# ------------------------------------------------------------------
def process_edf(edf_path: str, summary_path: str) -> tuple[np.ndarray, np.ndarray, list, int]:
    """Loads an EDF, extracts windows, and labels them based on the summary."""
    edf_filename = os.path.basename(edf_path)
    seizures = parse_seizures(summary_path, edf_filename)
    if not seizures:
        raise RuntimeError(f"No seizures parsed for {edf_filename}")

    print(f"\n[info] Processing {edf_filename}...")
    print(f"[info] Seizures parsed: {seizures}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    missing = [c for c in TARGET_CHANNELS if c not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing channels in {edf_filename}: {missing}")
        
    raw.pick_channels(TARGET_CHANNELS)
    data = raw.get_data()
    fs = int(raw.info["sfreq"])

    WIN = WIN_SEC * fs
    STRIDE = STRIDE_SEC * fs

    def window_label(start_sample: int) -> int:
        win_start_s = start_sample / fs
        win_end_s   = (start_sample + WIN) / fs
        for s, e in seizures:
            if win_end_s >= s and win_start_s <= e:
                return 1
        return 0

    X, y = [], []
    for start in range(0, data.shape[1] - WIN, STRIDE):
        window = data[:, start:start + WIN]
        X.append(_extract_features(window, fs=fs).flatten())
        y.append(window_label(start))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    
    print(f"[info] {edf_filename}: {len(y)} windows | Positives: {int(y.sum())} ({100*y.mean():.2f}%)")
    return X, y, seizures, fs


# ------------------------------------------------------------------
# 4. Execute Pipeline
# ------------------------------------------------------------------
if not os.path.exists(TRAIN_EDF) or not os.path.exists(TEST_EDF):
    raise FileNotFoundError("Make sure both EDF files are in data/data/")

# Process Train File
Xtr, ytr, seizures_train, fs = process_edf(TRAIN_EDF, SUMMARY_PATH)
if ytr.sum() == 0:
    raise RuntimeError("Train set has no positives!")

# Process Test File
Xte, yte, seizures_test, _ = process_edf(TEST_EDF, SUMMARY_PATH)
if yte.sum() == 0:
    raise RuntimeError("Test set has no positives!")

# ------------------------------------------------------------------
# 5. Train (with overfitting guards)
# ------------------------------------------------------------------
print("\n[info] Training Random Forest...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(Xtr, ytr)

# ------------------------------------------------------------------
# 6. Evaluate
# ------------------------------------------------------------------
y_pred = clf.predict(Xte)
y_prob = clf.predict_proba(Xte)[:, 1]

cm = confusion_matrix(yte, y_pred, labels=[0, 1])
print("\n[Confusion matrix]  rows=true, cols=pred")
print("              pred=0   pred=1")
print(f"  true=0      {cm[0, 0]:>6}   {cm[0, 1]:>6}")
print(f"  true=1      {cm[1, 0]:>6}   {cm[1, 1]:>6}")

print("\n[Classification report]")
print(classification_report(
    yte, y_pred, labels=[0, 1], target_names=["interictal", "seizure"], zero_division=0
))

print("\n[Threshold sweep]   prob >= t  ->  recall / false-alarm rate")
for t in [0.30, 0.40, 0.45, 0.50, 0.60, 0.75, 0.85, 0.90]:
    preds = (y_prob >= t).astype(int)
    tp = int(((preds == 1) & (yte == 1)).sum())
    fn = int(((preds == 0) & (yte == 1)).sum())
    fp = int(((preds == 1) & (yte == 0)).sum())
    tn = int(((preds == 0) & (yte == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr    = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"  p>={t:.2f}   recall={recall:.3f}   FPR={fpr:.3f}   (tp={tp} fn={fn} fp={fp} tn={tn})")

# ------------------------------------------------------------------
# 7. Save model + metadata
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)

meta = {
    "model_file":           os.path.basename(MODEL_PATH),
    "trained_at":           datetime.now(timezone.utc).isoformat(),
    "train_file":           os.path.basename(TRAIN_EDF),
    "test_file":            os.path.basename(TEST_EDF),
    "channels":             TARGET_CHANNELS,
    "fs":                   fs,
    "window_sec":           WIN_SEC,
    "stride_sec":           STRIDE_SEC,
    "n_features":           int(Xtr.shape[1]),
    "train_size":           int(len(ytr)),
    "test_size":            int(len(yte)),
    "seizures_train":       seizures_train,
    "seizures_test":        seizures_test,
    "model_params":         clf.get_params(),
    "sklearn_version":      sklearn.__version__,
    "python_version":       platform.python_version(),
    "notes": (
        "Cross-session POC: trained on chb01_03, tested on chb01_04. "
        "This is a legitimate holdout test for a single patient."
    ),
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2, default=str)

print(f"\n✅ Model    → {MODEL_PATH}")
print(f"✅ Metadata → {META_PATH}")



#save evaluation data for plotting later 

np.savez(

    "src/models/eval_results.npz",
    y_true=yte ,
    y_pred = y_pred , 
    y_prob=y_prob,
)
print(" Eval data to src/models/eval_results.npz")