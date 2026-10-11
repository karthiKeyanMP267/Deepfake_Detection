import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn.functional as F
from run_phase5_full_3v3_experiment import Fast3V3FusionModel
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd

ckpt = torch.load('experiments/05_full_3v3/models/best_model.pth', map_location='cpu')
model = Fast3V3FusionModel()
model.load_state_dict(ckpt['fusion_model_state_dict'])
model.eval()

test_features = torch.load('experiments/05_full_3v3/features/test_features.pt', map_location='cpu')
with torch.no_grad():
    outputs = model(
        efficientnet_features=test_features['efficientnet_features'],
        video_swin_features=test_features['video_swin_features'],
        face_mouth_features=test_features['face_mouth_features'],
        whisper_features=test_features['whisper_features'],
        wav2vec2_features=test_features['wav2vec2_features'],
        spectrogram_features=test_features['spectrogram_features'],
        audio_available=test_features['audio_available'],
        return_loss=False,
    )

logits = outputs['logits']
probs = F.softmax(logits, dim=1)[:, 1].numpy()
preds = (probs >= 0.5).astype(int)
targets = test_features['labels'].numpy()

print(f"Test Set Samples: {len(targets)}")
print(f"Accuracy: {accuracy_score(targets, preds):.4f}")
print(f"F1: {f1_score(targets, preds):.4f}")
print(classification_report(targets, preds, target_names=['Real (0)', 'Fake (1)'], digits=4))

# Load test manifest to see sample names and manipulation types
test_manifest = pd.read_csv('experiments/03_vision_swin/same_test_comparison/test_manifest.csv')

fn_indices = [i for i, (t, p) in enumerate(zip(targets, preds)) if t == 1 and p == 0]
print(f"\nTotal False Negatives (Ground truth FAKE, predicted REAL): {len(fn_indices)}")
for idx in fn_indices:
    row = test_manifest.iloc[idx] if idx < len(test_manifest) else {}
    sid = test_features['sample_ids'][idx]
    p_fake = probs[idx]
    manip = row.get('manipulation', 'Unknown')
    v_path = row.get('video_path', 'Unknown')
    print(f"  FN [{idx}]: sample_id={sid}, P(FAKE)={p_fake:.4f}, P(REAL)={1-p_fake:.4f}, manip={manip}")
    print(f"       file: {v_path}")

fp_indices = [i for i, (t, p) in enumerate(zip(targets, preds)) if t == 0 and p == 1]
print(f"\nTotal False Positives (Ground truth REAL, predicted FAKE): {len(fp_indices)}")
for idx in fp_indices:
    row = test_manifest.iloc[idx] if idx < len(test_manifest) else {}
    sid = test_features['sample_ids'][idx]
    p_fake = probs[idx]
    v_path = row.get('video_path', 'Unknown')
    print(f"  FP [{idx}]: sample_id={sid}, P(FAKE)={p_fake:.4f}, P(REAL)={1-p_fake:.4f}")
    print(f"       file: {v_path}")
