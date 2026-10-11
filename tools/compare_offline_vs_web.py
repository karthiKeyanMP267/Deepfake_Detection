"""
compare_offline_vs_web.py

Compare offline precomputed feature evaluation with live end-to-end web inference
on 20 test manifest samples across all 4 categories.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from run_phase5_full_3v3_experiment import Fast3V3FusionModel
from app.inference_service import InferenceService
from app.video_processor import VideoProcessor
from models.multimodal_model import MultimodalDeepfakeDetector

# 1. Load best checkpoint and fusion model
ckpt = torch.load('experiments/05_full_3v3/models/best_model.pth', map_location='cpu')
fusion_model = Fast3V3FusionModel()
fusion_model.load_state_dict(ckpt['fusion_model_state_dict'])
fusion_model.eval()

# 2. Load test cached features & test manifest
test_features = torch.load('experiments/05_full_3v3/features/test_features.pt', map_location='cpu')
test_manifest = pd.read_csv('experiments/03_vision_swin/same_test_comparison/test_manifest.csv')

# 3. Load full detector (web inference)
service = InferenceService.get_instance()
service.load_model()
web_model = service.model
web_processor = VideoProcessor()
device = service.device

print(f"{'Sample ID':<12} | {'Category':<8} | {'GT':<5} | {'Offline P(Fake)':<16} | {'Web P(Fake)':<12} | {'Offline Pred':<12} | {'Web Pred':<10} | {'Match?':<6}")
print("-" * 100)

temp_dir = Path("app/temp_uploads/compare_test")
temp_dir.mkdir(parents=True, exist_ok=True)

# Select 5 samples from each category: RV_RA, FV_FA, FV_RA, RV_FA
categories = ['RV_RA', 'FV_FA', 'FV_RA', 'RV_FA']
selected_indices = []
for cat in categories:
    cat_indices = test_manifest[test_manifest['manipulation'] == cat].index.tolist()[:5]
    selected_indices.extend(cat_indices)

matches_count = 0
total_tested = 0

for idx in selected_indices:
    row = test_manifest.iloc[idx]
    sid = test_features['sample_ids'][idx]
    gt = int(test_features['labels'][idx])
    cat = row.get('manipulation', 'N/A')
    video_path = Path(row['video_path'])

    # A) Offline cached forward pass
    with torch.no_grad():
        out_off = fusion_model(
            efficientnet_features=test_features['efficientnet_features'][idx:idx+1],
            video_swin_features=test_features['video_swin_features'][idx:idx+1],
            face_mouth_features=test_features['face_mouth_features'][idx:idx+1],
            whisper_features=test_features['whisper_features'][idx:idx+1],
            wav2vec2_features=test_features['wav2vec2_features'][idx:idx+1],
            spectrogram_features=test_features['spectrogram_features'][idx:idx+1],
            audio_available=test_features['audio_available'][idx:idx+1],
            return_loss=False,
        )
        p_off_fake = F.softmax(out_off['logits'], dim=1)[0, 1].item()
        pred_off = "FAKE" if p_off_fake >= 0.50 else "REAL"

    # B) Web live inference
    if not video_path.exists():
        print(f"{sid:<12} | {cat:<8} | {gt:<5} | {p_off_fake:<16.4f} | {'FILE MISSING':<12} | {pred_off:<12} | {'N/A':<10} | {'SKIP':<6}")
        continue

    images_tensor, waveform_tensor, audio_avail, meta = web_processor.process_video_and_audio(video_path, temp_dir)
    images_dev = images_tensor.to(device)
    waveform_dev = waveform_tensor.to(device) if audio_avail else None

    with torch.inference_mode():
        out_web = web_model(
            images=images_dev,
            waveform=waveform_dev,
            audio_available=audio_avail,
            return_loss=False,
        )
        p_web_fake = F.softmax(out_web['logits'], dim=1)[0, 1].item()
        pred_web = "FAKE" if p_web_fake >= 0.50 else "REAL"

    match = "YES" if pred_off == pred_web else "NO"
    if match == "YES":
        matches_count += 1
    total_tested += 1

    gt_str = "FAKE" if gt == 1 else "REAL"
    print(f"{sid:<12} | {cat:<8} | {gt_str:<5} | {p_off_fake:<16.4f} | {p_web_fake:<12.4f} | {pred_off:<12} | {pred_web:<10} | {match:<6}")

print("-" * 100)
print(f"Total Compared: {total_tested}, Agreement Rate: {matches_count}/{total_tested} ({matches_count/max(1, total_tested)*100:.1f}%)")
