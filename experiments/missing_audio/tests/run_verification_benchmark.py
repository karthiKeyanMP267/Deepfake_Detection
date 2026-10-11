"""
experiments/missing_audio/tests/run_verification_benchmark.py

Technical Verification & Quantitative Benchmark for Missing Audio Enhancement.
Evaluates 100 test samples in both Normal Multimodal Mode and Vision-Only Mode.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from models.multimodal_model import MultimodalDeepfakeDetector
from training.dataloader import get_test_loader


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Verification Benchmark on Device: {device}")

    ckpt_path = os.path.join("experiments", "weighted_sampler", "models", "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = MultimodalDeepfakeDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = get_test_loader("processed/test.csv", batch_size=1)

    # Collect 100 test samples
    samples = []
    for i, batch in enumerate(test_loader):
        if i >= 100:
            break
        samples.append(batch)

    print(f"Successfully loaded {len(samples)} test samples from processed/test.csv.")

    # 1. Benchmark Normal Multimodal Mode (Vision + Audio)
    labels = []
    preds_multimodal = []
    probs_multimodal = []
    times_multimodal = []

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for batch in samples:
        images = batch["images"].to(device)
        waveform = batch["waveform"].to(device)
        lbl = batch["label"].item()
        labels.append(lbl)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(images, waveform, audio_available=True)
        t1 = time.perf_counter()

        times_multimodal.append((t1 - t0) * 1000)
        probs = torch.softmax(output["logits"], dim=1)
        preds_multimodal.append(torch.argmax(probs, dim=1).item())
        probs_multimodal.append(probs[0, 1].item())

    mem_multimodal_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
    )

    # 2. Benchmark Vision-Only Mode (Audio Skipped / Missing)
    preds_vision = []
    probs_vision = []
    times_vision = []

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for batch in samples:
        images = batch["images"].to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(images, waveform=None, audio_available=False)
        t1 = time.perf_counter()

        times_vision.append((t1 - t0) * 1000)
        probs = torch.softmax(output["logits"], dim=1)
        preds_vision.append(torch.argmax(probs, dim=1).item())
        probs_vision.append(probs[0, 1].item())

    mem_vision_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
    )

    # 3. Compute Metrics
    acc_m = accuracy_score(labels, preds_multimodal)
    prec_m = precision_score(labels, preds_multimodal, zero_division=0)
    rec_m = recall_score(labels, preds_multimodal, zero_division=0)
    f1_m = f1_score(labels, preds_multimodal, zero_division=0)
    try:
        auc_m = roc_auc_score(labels, probs_multimodal)
    except Exception:
        auc_m = 0.0
    cm_m = confusion_matrix(labels, preds_multimodal, labels=[0, 1])

    acc_v = accuracy_score(labels, preds_vision)
    prec_v = precision_score(labels, preds_vision, zero_division=0)
    rec_v = recall_score(labels, preds_vision, zero_division=0)
    f1_v = f1_score(labels, preds_vision, zero_division=0)
    try:
        auc_v = roc_auc_score(labels, probs_vision)
    except Exception:
        auc_v = 0.0
    cm_v = confusion_matrix(labels, preds_vision, labels=[0, 1])

    avg_time_m = float(np.mean(times_multimodal))
    avg_time_v = float(np.mean(times_vision))

    results = {
        "samples_evaluated": len(samples),
        "multimodal": {
            "accuracy": float(acc_m),
            "precision": float(prec_m),
            "recall": float(rec_m),
            "f1": float(f1_m),
            "auc": float(auc_m),
            "confusion_matrix": cm_m.tolist(),
            "avg_time_ms": round(avg_time_m, 2),
            "peak_gpu_mem_mb": round(mem_multimodal_mb, 2),
        },
        "vision_only": {
            "accuracy": float(acc_v),
            "precision": float(prec_v),
            "recall": float(rec_v),
            "f1": float(f1_v),
            "auc": float(auc_v),
            "confusion_matrix": cm_v.tolist(),
            "avg_time_ms": round(avg_time_v, 2),
            "peak_gpu_mem_mb": round(mem_vision_mb, 2),
        },
        "deltas": {
            "accuracy": float(acc_v - acc_m),
            "precision": float(prec_v - prec_m),
            "recall": float(rec_v - rec_m),
            "f1": float(f1_v - f1_m),
            "auc": float(auc_v - auc_m),
            "time_reduction_ms": round(avg_time_m - avg_time_v, 2),
            "speedup_percent": round(((avg_time_m - avg_time_v) / avg_time_m) * 100, 2),
        },
    }

    out_file = os.path.join("experiments", "missing_audio", "logs", "benchmark_verification.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 70)
    print("BENCHMARK VERIFICATION RESULTS (100 SAMPLES):")
    print("=" * 70)
    print(f"Normal Multimodal Mode : Accuracy={acc_m:.4f}, Precision={prec_m:.4f}, Recall={rec_m:.4f}, F1={f1_m:.4f}, AUC={auc_m:.4f}")
    print(f"                         Avg Time={avg_time_m:.2f} ms/sample, Peak Mem={mem_multimodal_mb:.2f} MB")
    print(f"Vision-Only Mode       : Accuracy={acc_v:.4f}, Precision={prec_v:.4f}, Recall={rec_v:.4f}, F1={f1_v:.4f}, AUC={auc_v:.4f}")
    print(f"                         Avg Time={avg_time_v:.2f} ms/sample, Peak Mem={mem_vision_mb:.2f} MB")
    print(f"Deltas                 : Speedup={results['deltas']['speedup_percent']:.2f}% ({results['deltas']['time_reduction_ms']:.2f} ms saved per sample)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_benchmark()
