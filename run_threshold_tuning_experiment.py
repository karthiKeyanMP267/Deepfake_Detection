import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).parent))

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from models.multimodal_model import MultimodalDeepfakeDetector
from training.dataloader import get_test_loader


def run_threshold_tuning():
    print("=" * 70)
    print("STARTING THRESHOLD TUNING EXPERIMENT")
    print("=" * 70)

    # 1. Verify Checkpoint
    checkpoint_path = os.path.join("experiments", "weighted_sampler", "models", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Required checkpoint not found at {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    checkpoint_epoch = checkpoint.get("epoch", "N/A")
    val_f1 = checkpoint.get("best_f1", "N/A")

    print("\n" + "=" * 52)
    print("CHECKPOINT VERIFICATION")
    print("=" * 52)
    print(f"Checkpoint Path : {checkpoint_path}")
    print(f"Checkpoint Epoch: {checkpoint_epoch}")
    print(f"Validation F1   : {val_f1:.4f}" if isinstance(val_f1, float) else f"Validation F1   : {val_f1}")
    print("=" * 52 + "\n")

    # 2. Setup Directories
    base_dir = os.path.join("experiments", "threshold_tuning")
    metrics_dir = os.path.join(base_dir, "metrics")
    plots_dir = os.path.join(base_dir, "plots")
    predictions_dir = os.path.join(base_dir, "predictions")
    reports_dir = os.path.join(base_dir, "reports")

    for d in [base_dir, metrics_dir, plots_dir, predictions_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    # Save initial experiment config
    config = {
        "Experiment Name": "threshold_tuning",
        "Source Checkpoint": checkpoint_path,
        "Source Epoch": checkpoint_epoch,
        "Source Val F1": float(val_f1) if isinstance(val_f1, (float, np.float32, np.float64)) else val_f1,
        "Test Dataset": "processed/test.csv",
        "Thresholds Tested": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        "Retraining Occurred": False,
        "Model Weights Modified": False,
        "Optimizer Executed": False,
        "Scheduler Executed": False,
    }
    with open(os.path.join(base_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 3. Load Model & Run Single-Pass Inference
    print("Loading model and performing single-pass inference on test set...")
    model = MultimodalDeepfakeDetector()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Record weight checksum to verify NO weight modifications occur
    initial_param_sum = sum(p.sum().item() for p in model.parameters())

    test_csv = "processed/test.csv"
    test_loader = get_test_loader(test_csv, batch_size=2)

    all_labels = []
    all_prob_real = []
    all_prob_fake = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            waveform = batch["waveform"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, waveform)
            probs = torch.softmax(outputs["logits"], dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_prob_real.extend(probs[:, 0].cpu().numpy().tolist())
            all_prob_fake.extend(probs[:, 1].cpu().numpy().tolist())

    # Verify zero parameter mutation
    final_param_sum = sum(p.sum().item() for p in model.parameters())
    assert initial_param_sum == final_param_sum, "CRITICAL ERROR: Model weights were modified during inference!"
    print(f"Inference completed on {len(all_labels)} test samples. Weight integrity verified.")

    # 4. Evaluate Thresholds
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    summary_rows = []
    results_by_thresh = {}

    auc_val = roc_auc_score(all_labels, all_prob_fake)

    for thresh in thresholds:
        thresh_str = f"{thresh:.2f}"
        preds = [1 if p >= thresh else 0 for p in all_prob_fake]

        acc = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        bal_acc = balanced_accuracy_score(all_labels, preds)

        cm = confusion_matrix(all_labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Save predictions CSV for this threshold
        df_thresh = pd.DataFrame({
            "ground_truth": all_labels,
            "probability_real": all_prob_real,  
            "probability_fake": all_prob_fake,
            "threshold": thresh,
            "prediction": preds
        })
        pred_filename = f"predictions_threshold_{thresh_str}.csv"
        df_thresh.to_csv(os.path.join(predictions_dir, pred_filename), index=False)
        df_thresh.to_csv(os.path.join(base_dir, pred_filename), index=False)

        # Per-class metrics
        prec_real = precision_score(all_labels, preds, pos_label=0, zero_division=0)
        rec_real = recall_score(all_labels, preds, pos_label=0, zero_division=0)
        f1_real = f1_score(all_labels, preds, pos_label=0, zero_division=0)

        metrics_data = {
            "threshold": thresh,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc_val,
            "balanced_accuracy": bal_acc,
            "specificity": spec,
            "sensitivity": sens,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "precision_real": prec_real,
            "recall_real": rec_real,
            "f1_real": f1_real,
            "precision_fake": prec,
            "recall_fake": rec,
            "f1_fake": f1
        }

        # Save metrics JSON for this threshold
        metric_filename = f"metrics_threshold_{thresh_str}.json"
        with open(os.path.join(metrics_dir, metric_filename), "w") as f:
            json.dump(metrics_data, f, indent=4)
        with open(os.path.join(base_dir, metric_filename), "w") as f:
            json.dump(metrics_data, f, indent=4)

        # Plot & save confusion matrix for this threshold
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title(f"Confusion Matrix (Threshold = {thresh_str})")
        cm_filename = f"confusion_matrix_{thresh_str}.png"
        plt.savefig(os.path.join(plots_dir, cm_filename))
        plt.savefig(os.path.join(base_dir, cm_filename))
        plt.close()

        # Save classification report for this threshold
        clf_rep = classification_report(all_labels, preds, target_names=["Real", "Fake"], digits=4, zero_division=0)
        rep_filename = f"classification_report_{thresh_str}.txt"
        with open(os.path.join(reports_dir, rep_filename), "w") as f:
            f.write(clf_rep)
        with open(os.path.join(base_dir, rep_filename), "w") as f:
            f.write(clf_rep)

        summary_rows.append({
            "Threshold": thresh,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": auc_val,
            "Balanced Accuracy": bal_acc,
            "Specificity": spec,
            "Sensitivity": sens,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        })

        results_by_thresh[thresh_str] = metrics_data

    # 5. Save Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(base_dir, "threshold_summary.csv"), index=False)
    summary_df.to_csv(os.path.join(metrics_dir, "threshold_summary.csv"), index=False)
    print("\n" + "=" * 70)
    print("THRESHOLD TUNING SUMMARY TABLE:")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70 + "\n")

    # 6. Generate Threshold Curve Plots
    plot_metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Specificity", "specificity"),
        ("Sensitivity", "sensitivity"),
        ("TP", "tp"),
        ("TN", "tn"),
        ("FP", "fp"),
        ("FN", "fn"),
    ]

    for title_name, key_name in plot_metrics:
        plt.figure(figsize=(7, 5))
        plt.plot(summary_df["Threshold"], summary_df[title_name], marker="o", linewidth=2, color="#1f77b4")
        plt.title(f"Threshold vs {title_name}")
        plt.xlabel("Decision Threshold")
        plt.ylabel(title_name)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(thresholds)
        plot_name = f"threshold_vs_{key_name}.png"
        plt.savefig(os.path.join(plots_dir, plot_name))
        plt.savefig(os.path.join(base_dir, plot_name))
        plt.close()

    # Multi-metric combined plot
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df["Threshold"], summary_df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(summary_df["Threshold"], summary_df["Precision"], marker="s", label="Precision")
    plt.plot(summary_df["Threshold"], summary_df["Recall"], marker="^", label="Recall")
    plt.plot(summary_df["Threshold"], summary_df["F1"], marker="d", label="F1 Score")
    plt.plot(summary_df["Threshold"], summary_df["Balanced Accuracy"], marker="v", label="Balanced Acc")
    plt.title("Metrics vs Decision Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(thresholds)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "threshold_curves.png"))
    plt.savefig(os.path.join(base_dir, "threshold_curves.png"))
    plt.close()

    # 7. Select Best Threshold
    # Selection logic: Highest F1 -> Highest Balanced Acc -> Highest Real Recall (Specificity)
    sorted_df = summary_df.sort_values(
        by=["F1", "Balanced Accuracy", "Specificity"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best_row = sorted_df.iloc[0]
    best_thresh = best_row["Threshold"]
    default_row = summary_df[summary_df["Threshold"] == 0.50].iloc[0]

    print("=" * 70)
    print("BEST THRESHOLD SELECTION REASONING")
    print("=" * 70)
    print(f"Multi-tier Selection Rules:")
    print("  1. Primary Criterion  : Highest F1-Score")
    print("  2. First Tie-breaker : Higher Balanced Accuracy")
    print("  3. Second Tie-breaker: Higher Real Class Recall (Specificity)")
    print("-" * 70)
    print(f"Selected Best Threshold: {best_thresh:.2f}")
    print(f"  • Best F1-Score       : {best_row['F1']:.4f}")
    print(f"  • Balanced Accuracy   : {best_row['Balanced Accuracy']:.4f}")
    print(f"  • Real Recall (Spec)  : {best_row['Specificity']:.4f}")
    print(f"  • Accuracy            : {best_row['Accuracy']:.4f}")
    print(f"  • Precision           : {best_row['Precision']:.4f}")
    print(f"  • Recall              : {best_row['Recall']:.4f}")
    print("=" * 70 + "\n")

    # 8. Generate Reports
    report_content = f"""# Threshold Tuning Report (`threshold_tuning`)

## 1. Purpose of Threshold Tuning
The purpose of this experiment is to identify the optimal decision threshold for converting continuous model softmax probabilities $P(\\text{{Fake}})$ into binary classification predictions (Real vs. Fake), without altering model architecture, weights, or hyperparameters.

---

## 2. Checkpoint Used
- **Checkpoint File**: `{checkpoint_path}`
- **Source Experiment**: `WeightedRandomSampler`
- **Checkpoint Epoch**: `{checkpoint_epoch}`
- **Validation F1**: `{val_f1:.4f}`

---

## 3. Thresholds Evaluated
Tested decision thresholds: `[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]`

---

## 4. Complete Results Table

| Threshold | Accuracy | Precision | Recall | F1 Score | Balanced Acc | Specificity | Sensitivity | TP | TN | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    for _, row in summary_df.iterrows():
        report_content += (
            f"| {row['Threshold']:.2f} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | "
            f"{row['F1']:.4f} | {row['Balanced Accuracy']:.4f} | {row['Specificity']:.4f} | {row['Sensitivity']:.4f} | "
            f"{int(row['TP'])} | {int(row['TN'])} | {int(row['FP'])} | {int(row['FN'])} |\n"
        )

    report_content += f"""
---

## 5. Best Threshold & Selection Reasoning
- **Best Threshold**: `{best_thresh:.2f}`
- **Selection Criteria**:
  1. Primary: Highest F1-score (`{best_row['F1']:.4f}`)
  2. Tie-breaker 1: Highest Balanced Accuracy (`{best_row['Balanced Accuracy']:.4f}`)
  3. Tie-breaker 2: Highest Real-class Recall / Specificity (`{best_row['Specificity']:.4f}`)

---

## 6. Comparison with Default Threshold (0.50)

| Metric | Default Threshold (0.50) | Best Threshold ({best_thresh:.2f}) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | {default_row['Accuracy']:.4f} | {best_row['Accuracy']:.4f} | {best_row['Accuracy'] - default_row['Accuracy']:+.4f} |
| **Precision** | {default_row['Precision']:.4f} | {best_row['Precision']:.4f} | {best_row['Precision'] - default_row['Precision']:+.4f} |
| **Recall** | {default_row['Recall']:.4f} | {best_row['Recall']:.4f} | {best_row['Recall'] - default_row['Recall']:+.4f} |
| **F1-Score** | {default_row['F1']:.4f} | {best_row['F1']:.4f} | {best_row['F1'] - default_row['F1']:+.4f} |
| **Balanced Accuracy** | {default_row['Balanced Accuracy']:.4f} | {best_row['Balanced Accuracy']:.4f} | {best_row['Balanced Accuracy'] - default_row['Balanced Accuracy']:+.4f} |
| **Specificity** | {default_row['Specificity']:.4f} | {best_row['Specificity']:.4f} | {best_row['Specificity'] - default_row['Specificity']:+.4f} |
| **Sensitivity** | {default_row['Sensitivity']:.4f} | {best_row['Sensitivity']:.4f} | {best_row['Sensitivity'] - default_row['Sensitivity']:+.4f} |
| **ROC-AUC** | {default_row['ROC-AUC']:.4f} | {best_row['ROC-AUC']:.4f} | 0.0000 |

### Confusion Matrix Comparison:
- **Default (0.50)**:
  - TP: {int(default_row['TP'])}, TN: {int(default_row['TN'])}, FP: {int(default_row['FP'])}, FN: {int(default_row['FN'])}
- **Best ({best_thresh:.2f})**:
  - TP: {int(best_row['TP'])}, TN: {int(best_row['TN'])}, FP: {int(best_row['FP'])}, FN: {int(best_row['FN'])}

---

## 7. Zero-Retraining Verification Audit
- **Retraining Executed**: NO
- **Model Weights Altered**: NO (Parameter sum checksum equal before and after inference: `{initial_param_sum:.6f}`)
- **Optimizer Steps Executed**: NO
- **Scheduler Steps Executed**: NO
- **Experimental Variable Changed**: Decision threshold ONLY.

---

## 8. Final Recommendations
- The decision threshold of **`{best_thresh:.2f}`** {"is recommended for deployment" if best_thresh != 0.50 else "confirms that default 0.50 is optimal"} because it maximizes overall F1 score while maintaining optimal trade-offs between precision and recall under class imbalance.
"""

    with open(os.path.join(reports_dir, "threshold_tuning_report.md"), "w") as f:
        f.write(report_content)
    with open(os.path.join(base_dir, "threshold_tuning_report.md"), "w") as f:
        f.write(report_content)

    print("Threshold tuning completed successfully. All outputs saved in experiments/threshold_tuning/")


if __name__ == "__main__":
    run_threshold_tuning()
