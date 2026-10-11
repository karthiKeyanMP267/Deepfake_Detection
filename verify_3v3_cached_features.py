"""
verify_3v3_cached_features.py

Numerical Audit of Phase 5 Full 3V3 Precomputed Feature Caches:
- experiments/05_full_3v3/features/train_features.pt
- experiments/05_full_3v3/features/val_features.pt
- experiments/05_full_3v3/features/test_features.pt
"""

import os
import sys
import json
from pathlib import Path
import torch
import numpy as np

def audit_tensor(name, tensor):
    if tensor is None:
        return {
            "name": name,
            "status": "None",
            "shape": None,
            "dtype": None,
            "nan_count": 0,
            "pos_inf_count": 0,
            "neg_inf_count": 0,
            "invalid_percent": 0.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    if not isinstance(tensor, torch.Tensor):
        return {
            "name": name,
            "status": f"Not a tensor ({type(tensor)})",
            "length": len(tensor) if hasattr(tensor, '__len__') else None
        }

    t = tensor.float()
    nan_mask = torch.isnan(t)
    pos_inf_mask = torch.isposinf(t)
    neg_inf_mask = torch.isneginf(t)
    invalid_mask = nan_mask | pos_inf_mask | neg_inf_mask

    nan_count = int(nan_mask.sum().item())
    pos_inf_count = int(pos_inf_mask.sum().item())
    neg_inf_count = int(neg_inf_mask.sum().item())
    total_elements = t.numel()
    invalid_count = nan_count + pos_inf_count + neg_inf_count
    invalid_percent = (invalid_count / total_elements * 100.0) if total_elements > 0 else 0.0

    valid_elements = t[~invalid_mask]
    if valid_elements.numel() > 0:
        min_val = float(valid_elements.min().item())
        max_val = float(valid_elements.max().item())
        mean_val = float(valid_elements.mean().item())
        std_val = float(valid_elements.std().item())
    else:
        min_val, max_val, mean_val, std_val = None, None, None, None

    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "total_elements": total_elements,
        "nan_count": nan_count,
        "pos_inf_count": pos_inf_count,
        "neg_inf_count": neg_inf_count,
        "invalid_count": invalid_count,
        "invalid_percent": invalid_percent,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
    }

def audit_split(split_name, file_path):
    print("=" * 60)
    print(f"3V3 FEATURE CACHE NUMERICAL AUDIT — {split_name.upper()}")
    print(f"File: {file_path}")
    print("=" * 60)

    if not Path(file_path).exists():
        print(f"ERROR: File not found: {file_path}")
        return {"error": f"File not found: {file_path}"}

    data = torch.load(file_path, map_location="cpu")
    print(f"Available feature keys: {list(data.keys())}")
    
    split_results = {"file": str(file_path), "keys": list(data.keys()), "features": {}}

    branches = [
        ("EfficientNet", "efficientnet_features"),
        ("Video Swin", "video_swin_features"),
        ("Face/Mouth", "face_mouth_features"),
        ("Whisper", "whisper_features"),
        ("Wav2Vec2", "wav2vec2_features"),
        ("Spectrogram CNN", "spectrogram_features"),
        ("Labels", "labels"),
        ("Audio Available", "audio_available"),
    ]

    for label, key in branches:
        if key in data:
            tensor = data[key]
            res = audit_tensor(key, tensor)
            split_results["features"][key] = res
            print(f"\n{label} ({key}):")
            print(f"  Shape          : {res.get('shape')}")
            print(f"  Dtype          : {res.get('dtype')}")
            print(f"  NaN Count      : {res.get('nan_count')}")
            print(f"  +Inf Count     : {res.get('pos_inf_count')}")
            print(f"  -Inf Count     : {res.get('neg_inf_count')}")
            print(f"  Invalid %      : {res.get('invalid_percent'):.4f}%")
            if res.get('min') is not None:
                print(f"  Min            : {res.get('min'):.6f}")
                print(f"  Max            : {res.get('max'):.6f}")
                print(f"  Mean           : {res.get('mean'):.6f}")
                print(f"  Std            : {res.get('std'):.6f}")
        else:
            print(f"\n{label} ({key}): KEY MISSING IN DICT")
            split_results["features"][key] = {"status": "MISSING"}

    # Also inspect Sample IDs
    if "sample_ids" in data:
        s_ids = data["sample_ids"]
        print(f"\nSample IDs Count : {len(s_ids)}")
        print(f"First 3 Sample IDs: {s_ids[:3]}")
        print(f"Last 3 Sample IDs : {s_ids[-3:]}")
        split_results["sample_ids_count"] = len(s_ids)

    return split_results

def main():
    debug_dir = Path("experiments/05_full_3v3/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", "experiments/05_full_3v3/features/train_features.pt"),
        ("val", "experiments/05_full_3v3/features/val_features.pt"),
        ("test", "experiments/05_full_3v3/features/test_features.pt"),
    ]

    all_audits = {}
    for split_name, fpath in splits:
        audit_res = audit_split(split_name, fpath)
        all_audits[split_name] = audit_res

    # Save to debug/feature_audit.json
    out_json = debug_dir / "feature_audit.json"
    with open(out_json, "w") as f:
        json.dump(all_audits, f, indent=4)
    print(f"\nSaved numerical audit report to {out_json}")

if __name__ == "__main__":
    main()
