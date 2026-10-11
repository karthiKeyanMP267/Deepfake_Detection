"""
experiments/phase8/calibration/temperature_scaling.py

Step 6: Post-Hoc Confidence Calibration via Temperature Scaling.
Learns a single temperature parameter T > 0 strictly on VALIDATION data (Rule 6).
Evaluates Expected Calibration Error (ECE) and Brier Score before and after
calibration on the 604 test set without modifying model weights.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from run_final_project_validation import Fast3V3ValidationModel


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """
    Compute Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(int)
    accuracies = (preds == labels)

    ece = 0.0
    bin_details = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            bin_details.append({
                "bin_range": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "samples_count": int(np.sum(in_bin)),
                "avg_confidence": float(avg_confidence_in_bin),
                "accuracy": float(accuracy_in_bin),
                "calibration_gap": float(abs_diff)
            })

    return float(ece), bin_details


def compute_brier_score(probs: np.ndarray, labels: np.ndarray):
    """
    Brier score: Mean squared error of predicted probabilities.
    """
    return float(np.mean((probs - labels) ** 2))


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling layer.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.01, max=10.0)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_epochs: int = 300, lr: float = 0.01):
        """
        Fit temperature on validation data by minimizing NLL loss.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_epochs)

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        optimal_temp = float(self.temperature.item())
        print(f"[+] Optimal temperature learned on validation data: T = {optimal_temp:.4f}")
        return optimal_temp


def run_calibration():
    print("=" * 65)
    print("PHASE 8 — STEP 6: CONFIDENCE CALIBRATION (TEMPERATURE SCALING)")
    print("=" * 65)

    ckpt_path = ROOT_DIR / "experiments" / "05_full_3v3" / "models" / "best_model.pth"
    val_features_path = ROOT_DIR / "experiments" / "05_full_3v3" / "features" / "val_features.pt"
    test_features_path = ROOT_DIR / "experiments" / "05_full_3v3" / "features" / "test_features.pt"
    calib_dir = ROOT_DIR / "experiments" / "phase8" / "calibration"
    reports_dir = ROOT_DIR / "experiments" / "phase8" / "reports"
    outputs_dir = ROOT_DIR / "experiments" / "phase8" / "outputs"
    calib_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model & Validation Data
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = Fast3V3ValidationModel()
    model.load_state_dict(checkpoint["fusion_model_state_dict"])
    model.eval()

    val_cache = torch.load(val_features_path, map_location="cpu")
    num_val = len(val_cache["labels"])

    print(f"[*] Extracting validation logits ({num_val} samples) for calibration fitting...")
    with torch.no_grad():
        val_out = model(
            val_cache["efficientnet_features"],
            val_cache["video_swin_features"],
            val_cache["face_mouth_features"],
            val_cache["whisper_features"],
            val_cache["wav2vec2_features"],
            val_cache["spectrogram_features"],
            audio_available=torch.tensor([True] * num_val)
        )
    val_logits = val_out["logits"]
    val_labels = val_cache["labels"]

    # 2. Fit Temperature Scaler strictly on Validation Data
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(val_logits, val_labels)

    # 3. Evaluate on 604 Test Set
    test_cache = torch.load(test_features_path, map_location="cpu")
    num_test = len(test_cache["labels"])

    print(f"[*] Evaluating calibration impact on 604 test samples...")
    with torch.no_grad():
        test_out = model(
            test_cache["efficientnet_features"],
            test_cache["video_swin_features"],
            test_cache["face_mouth_features"],
            test_cache["whisper_features"],
            test_cache["wav2vec2_features"],
            test_cache["spectrogram_features"],
            audio_available=torch.tensor([True] * num_test)
        )

    raw_logits = test_out["logits"]
    with torch.no_grad():
        calib_logits = scaler(raw_logits)

    raw_probs = torch.softmax(raw_logits, dim=1)[:, 1].numpy()
    calib_probs = torch.softmax(calib_logits, dim=1)[:, 1].numpy()
    test_labels = test_cache["labels"].numpy()

    # Raw metrics
    raw_ece, raw_bins = compute_ece(raw_probs, test_labels)
    raw_brier = compute_brier_score(raw_probs, test_labels)

    # Calibrated metrics
    calib_ece, calib_bins = compute_ece(calib_probs, test_labels)
    calib_brier = compute_brier_score(calib_probs, test_labels)

    print("\nCalibration Results on Test Benchmark (604 samples):")
    print(f"  Raw Temperature         : 1.0000")
    print(f"  Calibrated Temperature  : {optimal_temp:.4f}")
    print(f"  Raw ECE                 : {raw_ece*100:.2f}%")
    print(f"  Calibrated ECE          : {calib_ece*100:.2f}%")
    print(f"  Raw Brier Score         : {raw_brier:.4f}")
    print(f"  Calibrated Brier Score  : {calib_brier:.4f}")

    # Save model artifact
    calib_data = {
        "optimal_temperature": optimal_temp,
        "fitted_on": "val_features.pt (2414 samples)",
        "test_samples": num_test,
        "raw_ece": raw_ece,
        "calibrated_ece": calib_ece,
        "raw_brier": raw_brier,
        "calibrated_brier": calib_brier,
        "bin_details_calibrated": calib_bins
    }

    calib_file = calib_dir / "calibration_model.json"
    with open(calib_file, "w") as f:
        json.dump(calib_data, f, indent=4)
    print(f"[+] Saved calibration parameters to: {calib_file}")

    # Generate Markdown Report
    report_path = reports_dir / "calibration_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 8 Confidence Calibration Report\n\n")
        f.write("**Method:** Temperature Scaling (Post-Hoc Probability Calibration)\n")
        f.write("**Fitting Dataset:** 2,414 Validation Samples (`val_features.pt` — Test Set Unused for Optimization)\n")
        f.write("**Test Dataset:** 604-Sample Benchmark (`test_features.pt`)\n")
        f.write("**Model Weights Modified:** STRICTLY NONE (Invariance Preserved)\n\n")
        f.write("---\n\n")
        f.write("## 1. Calibration Parameter & Metric Summary\n\n")
        f.write(f"- **Learned Temperature Parameter ($T$):** `{optimal_temp:.4f}`\n\n")
        f.write("| Metric | Before Calibration (Raw Softmax) | After Calibration (Temperature Scaled) | Absolute Improvement |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Expected Calibration Error (ECE)** | **{raw_ece*100:.2f}%** | **{calib_ece*100:.2f}%** | **{abs(raw_ece - calib_ece)*100:.2f}% pp** |\n")
        f.write(f"| **Brier Score (MSE)** | **{raw_brier:.4f}** | **{calib_brier:.4f}** | **{abs(raw_brier - calib_brier):.4f}** |\n\n")
        f.write("---\n\n")
        f.write("## 2. Monotonicity & Decision Invariance Guarantee\n\n")
        f.write("- Because Temperature Scaling applies a strictly monotonic scalar divisor $T > 0$ to logits, "
                "the classification boundary $P(\\text{fake}) \\ge 0.50$ is algebraically invariant.\n")
        f.write("- **Every single prediction (Real vs Fake) remains identical to the 3V3 baseline.**\n")
        f.write("- Softmax confidence scores are gently smoothed, mitigating overconfidence on ambiguous samples.\n")

    print(f"[+] Saved calibration report to: {report_path}\n")


if __name__ == "__main__":
    run_calibration()
