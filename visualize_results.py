"""
Generate comprehensive evaluation plots and structured metrics from saved predictions.
Run: python visualize_results.py
Outputs:
  - src/models/evaluation_report.png (clean multi-page figure)
  - src/models/evaluation_metrics.json (structured metrics)
  - src/models/threshold_analysis.csv (threshold sweep table)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# ------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------
data = np.load("src/models/eval_results.npz")
y_true = data["y_true"]
y_pred = data["y_pred"]
y_prob = data["y_prob"]

meta_path = Path("src/models/seizure_rf_model.meta.json")
meta = json.load(open(meta_path)) if meta_path.exists() else {}

# ------------------------------------------------------------------
# 2. Compute Core Metrics
# ------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

brier = brier_score_loss(y_true, y_prob)
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

clf_report = classification_report(
    y_true, y_pred, labels=[0, 1], target_names=["interictal", "seizure"],
    output_dict=True, zero_division=0
)

# ------------------------------------------------------------------
# 3. Threshold Sweep Table
# ------------------------------------------------------------------
thresholds = np.linspace(0, 1, 101)
rows = []
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    tp_t = ((preds == 1) & (y_true == 1)).sum()
    fn_t = ((preds == 0) & (y_true == 1)).sum()
    fp_t = ((preds == 1) & (y_true == 0)).sum()
    tn_t = ((preds == 0) & (y_true == 0)).sum()
    rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    f1_t = 2 * prec_t * rec / (prec_t + rec) if (prec_t + rec) > 0 else 0
    rows.append({
        "threshold": round(t, 3), "recall": round(rec, 4), "precision": round(prec_t, 4),
        "specificity": round(spec_t, 4), "f1": round(f1_t, 4), "fpr": round(fpr_t, 4),
        "tp": int(tp_t), "fn": int(fn_t), "fp": int(fp_t), "tn": int(tn_t),
    })
threshold_df = pd.DataFrame(rows)
key_thresholds = [0.30, 0.40, 0.45, 0.50, 0.60, 0.75, 0.85, 0.90]
key_rows = threshold_df[threshold_df["threshold"].isin(key_thresholds)]

# ------------------------------------------------------------------
# 4. Save Structured Outputs
# ------------------------------------------------------------------
out_dir = Path("src/models")
out_dir.mkdir(exist_ok=True)

metrics = {
    "dataset": {
        "train_file": meta.get("train_file", "unknown"),
        "test_file": meta.get("test_file", "unknown"),
        "n_windows": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence_pct": round(100 * y_true.mean(), 2),
    },
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "metrics_at_0.50": {
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "specificity": round(specificity, 4), "f1": round(f1, 4),
    },
    "threshold_independent": {
        "roc_auc": round(roc_auc, 4), "pr_auc": round(ap, 4), "brier_score": round(brier, 4),
    },
    "classification_report": clf_report,
    "key_operating_points": key_rows.to_dict(orient="records"),
    "calibration": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
}
with open(out_dir / "evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
threshold_df.to_csv(out_dir / "threshold_analysis.csv", index=False)

print("✅ Structured metrics saved → src/models/evaluation_metrics.json")
print("✅ Threshold table saved → src/models/threshold_analysis.csv")

# ------------------------------------------------------------------
# 5. Visualization — Clean, Separate Figures
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {"seizure": "#e74c3c", "interictal": "#3498db", "accent": "#f39c12", "dark": "#2c3e50"}

# ================================================================
# FIGURE 1: Core Performance (2x2)
# ================================================================
fig1, axes = plt.subplots(2, 2, figsize=(13, 10))
fig1.suptitle("Seizure Detection — Cross-Session Evaluation (chb01_03 → chb01_04)",
              fontsize=15, fontweight="bold", y=0.98)

# 1A: Confusion Matrix (counts)
ax = axes[0, 0]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Interictal", "Seizure"], yticklabels=["Interictal", "Seizure"],
            cbar_kws={"label": "Count"}, annot_kws={"size": 14})
ax.set_title("Confusion Matrix (Threshold = 0.50)", pad=12)
ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
ax.text(0.5, -0.15, f"Accuracy={accuracy:.3f}  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}",
        transform=ax.transAxes, ha="center", fontsize=10, color="gray")

# 1B: Normalized Confusion Matrix
ax = axes[0, 1]
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Greens", ax=ax,
            xticklabels=["Interictal", "Seizure"], yticklabels=["Interictal", "Seizure"],
            cbar_kws={"label": "Proportion"}, annot_kws={"size": 14})
ax.set_title("Normalized Confusion Matrix (Row %)", pad=12)
ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
ax.text(0.5, -0.15, f"Sensitivity={recall:.1%}  Specificity={specificity:.1%}",
        transform=ax.transAxes, ha="center", fontsize=10, color="gray")

# 1C: ROC Curve
ax = axes[1, 0]
ax.plot(fpr, tpr, color=PALETTE["dark"], lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1.5, ls="--", label="Random (AUC = 0.5)")
ax.fill_between(fpr, tpr, alpha=0.1, color=PALETTE["dark"])
for t in key_thresholds:
    row = threshold_df[threshold_df["threshold"] == t].iloc[0]
    ax.plot(row["fpr"], row["recall"], "o", color=PALETTE["seizure"], ms=5)
    ax.annotate(f"{t:.2f}", (row["fpr"], row["recall"]), xytext=(3, 3),
                textcoords="offset points", fontsize=8, color=PALETTE["seizure"])
ax.set_title("ROC Curve", pad=12)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

# 1D: Precision-Recall Curve
ax = axes[1, 1]
ax.plot(recall_curve, precision_curve, color=PALETTE["accent"], lw=2.5, label=f"PR (AP = {ap:.3f})")
ax.axhline(y=y_true.mean(), color="gray", lw=1.5, ls="--", label=f"Baseline (prev={y_true.mean():.3f})")
ax.fill_between(recall_curve, precision_curve, alpha=0.1, color=PALETTE["accent"])
ax.set_title("Precision-Recall Curve", pad=12)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(loc="lower left", fontsize=9)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.05)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig(out_dir / "evaluation_report.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig1)

# ================================================================
# FIGURE 2: Threshold Analysis (1x3)
# ================================================================
fig2, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig2.suptitle("Threshold Analysis", fontsize=14, fontweight="bold", y=1.02)

# 2A: Recall / Precision / FPR vs Threshold
ax = axes[0]
ax.plot(thresholds, threshold_df["recall"], label="Recall (Sensitivity)", color=PALETTE["seizure"], lw=2.5)
ax.plot(thresholds, threshold_df["precision"], label="Precision", color=PALETTE["interictal"], lw=2)
ax.plot(thresholds, threshold_df["fpr"], label="False Positive Rate", color="#95a5a6", lw=2)
for t in key_thresholds:
    row = threshold_df[threshold_df["threshold"] == t].iloc[0]
    ax.plot(t, row["recall"], "o", color=PALETTE["seizure"], ms=4)
ax.axvline(0.50, color="black", ls="--", alpha=0.4)
ax.axvline(0.40, color=PALETTE["accent"], ls=":", alpha=0.7, label="Clinical OP (0.40)")
ax.set_title("Recall / Precision / FPR vs Threshold")
ax.set_xlabel("Probability Threshold"); ax.set_ylabel("Rate")
ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=9, loc="lower left")

# 2B: F1 / Precision / Recall vs Threshold
ax = axes[1]
ax.plot(thresholds, threshold_df["f1"], label="F1-Score", color=PALETTE["accent"], lw=2.5)
ax.plot(thresholds, threshold_df["precision"], label="Precision", color=PALETTE["interictal"], lw=2)
ax.plot(thresholds, threshold_df["recall"], label="Recall", color=PALETTE["seizure"], lw=2)
max_f1_idx = threshold_df["f1"].idxmax()
ax.plot(threshold_df.loc[max_f1_idx, "threshold"], threshold_df.loc[max_f1_idx, "f1"],
        "*", color=PALETTE["accent"], ms=14,
        label=f"Max F1 @ {threshold_df.loc[max_f1_idx, 'threshold']:.2f} ({threshold_df.loc[max_f1_idx, 'f1']:.3f})")
ax.axvline(0.50, color="black", ls="--", alpha=0.4)
ax.axvline(0.40, color=PALETTE["accent"], ls=":", alpha=0.7)
ax.set_title("F1 / Precision / Recall vs Threshold")
ax.set_xlabel("Probability Threshold"); ax.set_ylabel("Score")
ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=9, loc="lower left")

# 2C: Key Operating Points Table
ax = axes[2]
ax.axis("off")
table_data = key_rows[["threshold", "recall", "precision", "specificity", "f1", "fpr", "tp", "fp"]].copy()
table_data.columns = ["Thresh", "Recall", "Prec", "Spec", "F1", "FPR", "TP", "FP"]
table_data = table_data.round({"Recall": 3, "Prec": 3, "Spec": 3, "F1": 3, "FPR": 4})
tbl = ax.table(cellText=table_data.values, colLabels=table_data.columns,
               cellLoc="center", loc="center", colColours=["#ecf0f1"]*8)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight="bold", color="white")
        cell.set_facecolor(PALETTE["dark"])
    elif r % 2 == 0:
        cell.set_facecolor("#f8f9fa")
ax.set_title("Key Operating Points", pad=10, fontweight="bold")

plt.tight_layout()
fig2.savefig(out_dir / "threshold_analysis.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig2)

# ================================================================
# FIGURE 3: Probability Distributions & Calibration (1x2)
# ================================================================
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("Probability Distributions & Calibration", fontsize=14, fontweight="bold", y=1.02)

# 3A: Probability Histograms
ax = axes[0]
bins = np.linspace(0, 1, 51)
ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.55, label="Interictal (True=0)",
        color=PALETTE["interictal"], density=True, edgecolor="white", linewidth=0.3)
ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.75, label="Seizure (True=1)",
        color=PALETTE["seizure"], density=True, edgecolor="white", linewidth=0.3)
ax.axvline(0.50, color="black", ls="--", alpha=0.5, label="Threshold=0.50")
ax.axvline(0.40, color=PALETTE["accent"], ls=":", alpha=0.7, label="Clinical OP=0.40")
ax.set_title("Predicted Probability by True Class (log density)")
ax.set_xlabel("Predicted Probability (Seizure)"); ax.set_ylabel("Density (log)")
ax.set_yscale("log"); ax.set_ylim(1e-3, 1e2)
ax.legend(fontsize=9)

# 3B: Calibration Curve
ax = axes[1]
ax.plot([0, 1], [0, 1], "k:", lw=1.5, label="Perfectly Calibrated")
ax.plot(prob_pred, prob_true, "s-", color=PALETTE["dark"], lw=2, ms=7, label="Model")
ax.set_title("Calibration Curve (Reliability Diagram)")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.legend(fontsize=9)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

plt.tight_layout()
fig3.savefig(out_dir / "probability_calibration.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig3)

print("✅ Core plot → src/models/evaluation_report.png")
print("✅ Threshold plot → src/models/threshold_analysis.png")
print("✅ Probability/Calibration plot → src/models/probability_calibration.png")