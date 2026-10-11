"""
experiments/phase8/verify_baseline.py

Step 2: Non-Destructive Baseline Reproduction Verification Script.
Evaluates the existing 3V3 champion checkpoint on the exact 604 test samples
to confirm baseline metric consistency before executing any Phase 8 modules.
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    classification_report
)

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from run_final_project_validation import Fast3V3ValidationModel


def run_verification():
    print("=" * 65)
    print("PHASE 8 — STEP 2: BASELINE REPRODUCTION VERIFICATION")
    print("=" * 65)

    ckpt_path = ROOT_DIR / "experiments" / "05_full_3v3" / "models" / "best_model.pth"
    features_path = ROOT_DIR / "experiments" / "05_full_3v3" / "features" / "test_features.pt"
    outputs_dir = ROOT_DIR / "experiments" / "phase8" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Champion checkpoint missing at: {ckpt_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Test features cache missing at: {features_path}")

    # 1. Load Checkpoint
    print(f"[*] Loading champion weights from: {ckpt_path.name}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    fusion_state_dict = checkpoint["fusion_model_state_dict"]

    model = Fast3V3ValidationModel()
    model.load_state_dict(fusion_state_dict)
    model.eval()

    # 2. Load Precomputed 604 Test Features
    print(f"[*] Loading test cache from: {features_path.name}...")
    test_cache = torch.load(features_path, map_location="cpu")
    labels = test_cache["labels"].numpy()
    num_samples = len(labels)
    assert num_samples == 604, f"Expected 604 samples, found {num_samples}!"
    assert (labels == 0).sum() == 19, f"Expected 19 Real samples, found {(labels == 0).sum()}!"
    assert (labels == 1).sum() == 585, f"Expected 585 Fake samples, found {(labels == 1).sum()}!"

    # 3. Execute Inference Pass
    print("[*] Executing forward inference on 604 test representations...")
    with torch.no_grad():
        out = model(
            test_cache["efficientnet_features"],
            test_cache["video_swin_features"],
            test_cache["face_mouth_features"],
            test_cache["whisper_features"],
            test_cache["wav2vec2_features"],
            test_cache["spectrogram_features"],
            audio_available=torch.tensor([True] * num_samples)
        )

    logits = out["logits"]
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

    # 4. Compute Metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    bal_acc = balanced_accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    f1_real = f1_score(labels, preds, pos_label=0)
    f1_fake = f1_score(labels, preds, pos_label=1)

    print("\n" + "-" * 50)
    print("VERIFICATION REPRODUCTION RESULTS:")
    print("-" * 50)
    print(f"  Accuracy:          {acc*100:.2f}% (Expected: ~97.52%)")
    print(f"  F1 Score:          {f1*100:.2f}% (Expected: ~98.71%)")
    print(f"  ROC-AUC:           {auc*100:.2f}% (Expected: ~99.02%)")
    print(f"  Balanced Accuracy: {bal_acc*100:.2f}% (Expected: ~93.63%)")
    print(f"  Real F1 Score:     {f1_real*100:.2f}% (Expected: ~69.39%)")
    print(f"  Fake F1 Score:     {f1_fake*100:.2f}% (Expected: ~98.71%)")
    print(f"  Specificity:       {spec*100:.2f}% (Expected: ~89.47%)")
    print(f"  Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("-" * 50)

    # 5. Assert Accuracy Tolerance
    expected_acc = 0.97516556
    expected_f1 = 0.9870578
    assert abs(acc - expected_acc) < 1e-4, f"Accuracy mismatch! Got {acc}, expected {expected_acc}"
    assert abs(f1 - expected_f1) < 1e-4, f"F1 mismatch! Got {f1}, expected {expected_f1}"
    print("[+] SUCCESS: Baseline metrics reproduced with 100% exact numerical fidelity!\n")

    # 6. Save reproduction artifact
    results = {
        "status": "PASSED",
        "sample_count": int(num_samples),
        "real_count": int((labels == 0).sum()),
        "fake_count": int((labels == 1).sum()),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(auc),
        "balanced_accuracy": float(bal_acc),
        "specificity": float(spec),
        "f1_real": float(f1_real),
        "f1_fake": float(f1_fake),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }

    out_file = outputs_dir / "baseline_reproduction.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[+] Saved baseline reproduction log to: {out_file}")
    return results


if __name__ == "__main__":
    run_verification()
