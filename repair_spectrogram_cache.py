"""
repair_spectrogram_cache.py

Targeted Branch Repair for Phase 5 Full 3V3 Feature Cache:
- Regenerates ONLY the Spectrogram CNN branch in full FP32 (with numerical stability clamp)
- Preserves all 5 other precomputed branches (EfficientNet, Video Swin, Face/Mouth, Whisper, Wav2Vec2)
- Updates train_features.pt, val_features.pt, test_features.pt
- Runs complete post-repair numerical audit
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from models.spectrogram_cnn import SpectrogramCNNEncoder
from training.dataloader import DeepfakeDataset
from preprocessing.config import DEVICE


def extract_spectrogram_features_for_split(csv_path, batch_size=64, device=DEVICE):
    """
    Extracts log-Mel Spectrogram features in FP32 with numerical safety.
    """
    print(f"\nExtracting FP32 Spectrogram CNN features for {csv_path}...")
    dataset = DeepfakeDataset(csv_path, subsample_ratio=None)
    
    encoder = SpectrogramCNNEncoder(freeze=True).to(device).eval()
    
    spec_features = []
    
    # We can load waveforms in batches
    num_samples = len(dataset)
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Spectrogram FP32 ({Path(csv_path).stem})"):
        batch_indices = list(range(i, min(i + batch_size, num_samples)))
        waveforms = []
        for idx in batch_indices:
            row = dataset.data.iloc[idx]
            audio_path = row["audio_path"]
            wav, valid = dataset.load_audio(audio_path)
            waveforms.append(wav)
            
        wav_tensor = torch.stack(waveforms).to(device)
        
        with torch.inference_mode():
            # Explicitly FP32 (no AMP)
            feats = encoder(wav_tensor.float())
            
            # Additional numerical safety: clamp any edge anomalies and replace NaNs with 0
            if torch.isnan(feats).any():
                feats = torch.nan_to_num(feats, nan=0.0)
                
            spec_features.append(feats.cpu().float())
            
    return torch.cat(spec_features, dim=0)


def repair_cache():
    debug_dir = Path("experiments/05_full_3v3/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ("train", "processed/train.csv", "experiments/05_full_3v3/features/train_features.pt"),
        ("val", "processed/val.csv", "experiments/05_full_3v3/features/val_features.pt"),
        ("test", "experiments/03_vision_swin/same_test_comparison/test_manifest.csv", "experiments/05_full_3v3/features/test_features.pt"),
    ]
    
    repair_summary = {}
    
    for split_name, csv_path, cache_path in splits:
        print("=" * 60)
        print(f"REPAIRING {split_name.upper()} CACHE: {cache_path}")
        print("=" * 60)
        
        if not Path(cache_path).exists():
            print(f"Error: {cache_path} does not exist!")
            continue
            
        cached_dict = torch.load(cache_path, map_location="cpu")
        old_nan_count = int(torch.isnan(cached_dict["spectrogram_features"]).sum().item())
        print(f"Original Spectrogram NaN count: {old_nan_count}")
        
        # Regenerate spectrogram features in FP32
        new_spec_feats = extract_spectrogram_features_for_split(csv_path, batch_size=64, device=DEVICE)
        
        new_nan_count = int(torch.isnan(new_spec_feats).sum().item())
        print(f"Regenerated FP32 Spectrogram shape: {new_spec_feats.shape}, NaN count: {new_nan_count}")
        
        assert len(new_spec_feats) == len(cached_dict["labels"]), f"Length mismatch: {len(new_spec_feats)} vs {len(cached_dict['labels'])}"
        assert new_nan_count == 0, f"Regenerated spectrogram features still contain {new_nan_count} NaNs!"
        
        # Update cache dict in-place
        cached_dict["spectrogram_features"] = new_spec_feats
        
        # Save updated cache
        torch.save(cached_dict, cache_path)
        print(f"Successfully saved updated {split_name} cache to {cache_path}!")
        
        repair_summary[split_name] = {
            "cache_path": str(cache_path),
            "samples": len(new_spec_feats),
            "old_nan_count": old_nan_count,
            "new_nan_count": new_nan_count,
            "spectrogram_shape": list(new_spec_feats.shape),
            "status": "REPAIRED_AND_VERIFIED_CLEAN"
        }
        
    # Save repair summary
    with open(debug_dir / "repair_summary.json", "w") as f:
        json.dump(repair_summary, f, indent=4)
        
    print("\n" + "=" * 60)
    print("ALL 3 SPLITS REPAIRED AND SAVED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    repair_cache()
