"""
experiments/phase8/ood/reliability_detector.py

Step 5: OOD & Forensic Reliability Detector.
Fits reference representations strictly on validation data (Rule 6: Never on test data).
Calculates embedding distance from reference prototypes, modality disagreement,
and quality indicators to produce HIGH / MEDIUM / LOW reliability tiers.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from run_final_project_validation import Fast3V3ValidationModel
from app.input_quality import InputQualityAnalyzer


class OODReliabilityDetector:
    """
    Forensic Reliability & Out-of-Distribution Detector.
    Strictly preserves the classifier's verdict while quantifying result trustworthiness.
    """

    CONSISTENCY_THRESHOLD = 0.1177  # Phase-6 validated threshold

    def __init__(self, stats_path: Optional[Path] = None):
        self.stats_path = stats_path or (ROOT_DIR / "experiments" / "phase8" / "ood" / "reference_stats.pt")
        self.reference_stats: Optional[Dict[str, Any]] = None
        if self.stats_path.exists():
            self.reference_stats = torch.load(self.stats_path, map_location="cpu")

    def fit_reference_distributions(
        self,
        val_features_path: Path,
        ckpt_path: Path,
    ) -> Dict[str, Any]:
        """
        Fit reference distribution statistics strictly on VALIDATION data (Rule 6).
        Computes class-conditional embedding centroids and variances.
        """
        print(f"[*] Fitting OOD reference distribution on validation data: {val_features_path.name}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model = Fast3V3ValidationModel()
        model.load_state_dict(checkpoint["fusion_model_state_dict"])
        model.eval()

        val_cache = torch.load(val_features_path, map_location="cpu")
        labels = val_cache["labels"].numpy()
        num_val = len(labels)

        with torch.no_grad():
            out = model(
                val_cache["efficientnet_features"],
                val_cache["video_swin_features"],
                val_cache["face_mouth_features"],
                val_cache["whisper_features"],
                val_cache["wav2vec2_features"],
                val_cache["spectrogram_features"],
                audio_available=torch.tensor([True] * num_val)
            )

        fused = out["fused_embedding"]  # Shape: (N, 512)
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        fused_real = fused[real_mask]
        fused_fake = fused[fake_mask]

        centroid_real = fused_real.mean(dim=0)
        centroid_fake = fused_fake.mean(dim=0)
        var_real = fused_real.var(dim=0, unbiased=False) + 1e-6
        var_fake = fused_fake.var(dim=0, unbiased=False) + 1e-6

        # Standardized Euclidean distance (diagonal Mahalanobis)
        def dist_to_centroid(feats, mu, var):
            diff = feats - mu.unsqueeze(0)
            return torch.sqrt((diff ** 2 / var.unsqueeze(0)).sum(dim=-1))

        d_real = dist_to_centroid(fused, centroid_real, var_real)
        d_fake = dist_to_centroid(fused, centroid_fake, var_fake)
        d_min_val = torch.minimum(d_real, d_fake).numpy()

        # In-distribution 95th and 99th percentile thresholds
        thresh_95 = float(np.percentile(d_min_val, 95))
        thresh_99 = float(np.percentile(d_min_val, 99))

        stats = {
            "centroid_real": centroid_real,
            "centroid_fake": centroid_fake,
            "var_real": var_real,
            "var_fake": var_fake,
            "dist_thresh_95": thresh_95,
            "dist_thresh_99": thresh_99,
            "val_samples": int(num_val),
            "val_real_count": int(real_mask.sum()),
            "val_fake_count": int(fake_mask.sum()),
        }

        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, self.stats_path)
        self.reference_stats = stats
        print(f"[+] Saved OOD reference stats to: {self.stats_path}")
        print(f"    Validation in-distribution distance 95th-pct: {thresh_95:.2f}, 99th-pct: {thresh_99:.2f}")
        return stats

    def evaluate_reliability(
        self,
        fused_embedding: torch.Tensor,
        prediction: str,
        confidence: float,
        audio_available: bool,
        vision_weight: float,
        audio_weight: float,
        av_similarity: Optional[float] = None,
        input_quality: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize embedding distance, modality weights, AV consistency, and input quality
        into a definitive reliability assessment.
        
        IMPORTANT: NEVER changes prediction (Rule 7).
        """
        ood_flags = []
        reliability_penalty = 0.0

        # 1. Feature Space Distance to Training Prototypes
        embedding_distance = 0.0
        is_embedding_ood = False
        if self.reference_stats is not None:
            f = fused_embedding.detach().cpu()
            if f.ndim == 1:
                f = f.unsqueeze(0)
            mu_r = self.reference_stats["centroid_real"].unsqueeze(0)
            var_r = self.reference_stats["var_real"].unsqueeze(0)
            mu_f = self.reference_stats["centroid_fake"].unsqueeze(0)
            var_f = self.reference_stats["var_fake"].unsqueeze(0)

            d_r = float(torch.sqrt(((f - mu_r) ** 2 / var_r).sum(dim=-1)).item())
            d_f = float(torch.sqrt(((f - mu_f) ** 2 / var_f).sum(dim=-1)).item())
            embedding_distance = round(min(d_r, d_f), 2)

            t95 = self.reference_stats["dist_thresh_95"]
            t99 = self.reference_stats["dist_thresh_99"]

            if embedding_distance > t99:
                is_embedding_ood = True
                ood_flags.append(f"Representation distance ({embedding_distance:.1f}) exceeds in-distribution 99th-percentile ({t99:.1f}).")
                reliability_penalty += 0.35
            elif embedding_distance > t95:
                is_embedding_ood = True
                ood_flags.append(f"Representation distance ({embedding_distance:.1f}) exceeds in-distribution 95th-percentile ({t95:.1f}).")
                reliability_penalty += 0.20

        # 2. Input Quality Indicators
        if input_quality is not None:
            video_props = input_quality.get("video", {})
            face_ratio = video_props.get("face_area_ratio", 0.5)
            is_screen_rec = video_props.get("is_widescreen_desktop", False)
            sharpness = video_props.get("frame_sharpness", 100.0)

            if is_screen_rec:
                ood_flags.append("Desktop screen recording format detected (multi-layer codec artifact smoothing).")
                reliability_penalty += 0.35
            if face_ratio > 0 and face_ratio < 0.15:
                ood_flags.append(f"Small facial ROI ({face_ratio*100:.1f}% frame area) suppresses high-frequency spatial artifact detection.")
                reliability_penalty += 0.30
            elif face_ratio > 0 and face_ratio < 0.25:
                reliability_penalty += 0.10
            if sharpness < 45.0:
                ood_flags.append("Severe frame defocus or blur detected.")
                reliability_penalty += 0.15

        # 3. Audio & Cross-Modal Consistency
        if not audio_available:
            reliability_penalty += 0.15
        elif av_similarity is not None and av_similarity < self.CONSISTENCY_THRESHOLD:
            ood_flags.append(f"Cross-modal desynchronization detected (S_AV {av_similarity:.4f} < {self.CONSISTENCY_THRESHOLD:.4f}).")
            reliability_penalty += 0.10

        # 4. Synthesize Reliability Tier
        reliability_score = max(0.10, min(1.0, round(1.0 - reliability_penalty, 2)))

        is_ood = is_embedding_ood or len(ood_flags) > 0

        if reliability_score >= 0.75 and len(ood_flags) == 0:
            reliability_tier = "HIGH"
            ood_status = "IN-DISTRIBUTION"
            explanation = "Input aligns closely with verified benchmark training distribution standards."
        elif reliability_score >= 0.50 and not (is_embedding_ood or (input_quality and input_quality.get("video", {}).get("is_widescreen_desktop", False))):
            reliability_tier = "MEDIUM"
            ood_status = "IN-DISTRIBUTION"
            explanation = "Input displays minor stream variances (e.g. vision-only mode or mild compression), but remains reliably interpretable."
        else:
            reliability_tier = "LOW"
            ood_status = "POSSIBLY OOD"
            explanation = (
                f"The detector produced a {prediction} prediction, but the input differs from conditions "
                "represented in the training data (e.g. screen recording, small face ROI, or atypical feature distribution). "
                "Therefore, the prediction should be treated with low reliability."
            )

        return {
            "verdict": prediction,          # NEVER CHANGED
            "confidence": confidence,      # NEVER CHANGED
            "reliability_tier": reliability_tier,
            "reliability_score": reliability_score,
            "ood_status": ood_status,
            "is_ood": is_ood,
            "embedding_distance": embedding_distance,
            "ood_flags": ood_flags,
            "explanation": explanation,
        }


def run_ood_evaluation():
    print("=" * 65)
    print("PHASE 8 — STEP 5: OOD & RELIABILITY EVALUATION")
    print("=" * 65)

    ckpt_path = ROOT_DIR / "experiments" / "05_full_3v3" / "models" / "best_model.pth"
    val_features_path = ROOT_DIR / "experiments" / "05_full_3v3" / "features" / "val_features.pt"
    test_features_path = ROOT_DIR / "experiments" / "05_full_3v3" / "features" / "test_features.pt"
    reports_dir = ROOT_DIR / "experiments" / "phase8" / "reports"
    outputs_dir = ROOT_DIR / "experiments" / "phase8" / "outputs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    detector = OODReliabilityDetector()
    stats = detector.fit_reference_distributions(val_features_path, ckpt_path)

    # Evaluate on the 604 test set (In-Distribution benchmark)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = Fast3V3ValidationModel()
    model.load_state_dict(checkpoint["fusion_model_state_dict"])
    model.eval()

    test_cache = torch.load(test_features_path, map_location="cpu")
    num_test = len(test_cache["labels"])

    with torch.no_grad():
        out = model(
            test_cache["efficientnet_features"],
            test_cache["video_swin_features"],
            test_cache["face_mouth_features"],
            test_cache["whisper_features"],
            test_cache["wav2vec2_features"],
            test_cache["spectrogram_features"],
            audio_available=torch.tensor([True] * num_test)
        )

    probs = torch.softmax(out["logits"], dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    fused_embs = out["fused_embedding"]
    vis_w = out["vision_weight"].numpy()
    aud_w = out["audio_weight"].numpy()
    av_sim = out["av_similarity"].numpy()

    high_cnt, med_cnt, low_cnt = 0, 0, 0
    in_dist_cnt, ood_cnt = 0, 0

    for i in range(num_test):
        p_str = "FAKE" if preds[i] == 1 else "REAL"
        conf = float(probs[i] if preds[i] == 1 else 1.0 - probs[i])
        res = detector.evaluate_reliability(
            fused_embedding=fused_embs[i],
            prediction=p_str,
            confidence=conf,
            audio_available=True,
            vision_weight=float(vis_w[i]),
            audio_weight=float(aud_w[i]),
            av_similarity=float(av_sim[i]),
            input_quality=None
        )
        if res["reliability_tier"] == "HIGH":
            high_cnt += 1
        elif res["reliability_tier"] == "MEDIUM":
            med_cnt += 1
        else:
            low_cnt += 1

        if res["ood_status"] == "IN-DISTRIBUTION":
            in_dist_cnt += 1
        else:
            ood_cnt += 1

    print("\nBenchmark Test Set (604 samples) Reliability Distribution:")
    print(f"  HIGH Reliability  : {high_cnt} ({high_cnt/num_test*100:.1f}%)")
    print(f"  MEDIUM Reliability: {med_cnt} ({med_cnt/num_test*100:.1f}%)")
    print(f"  LOW Reliability   : {low_cnt} ({low_cnt/num_test*100:.1f}%)")
    print(f"  IN-DISTRIBUTION   : {in_dist_cnt} ({in_dist_cnt/num_test*100:.1f}%)")
    print(f"  POSSIBLY OOD      : {ood_cnt} ({ood_cnt/num_test*100:.1f}%)")

    ood_eval_results = {
        "validation_samples_fitted": stats["val_samples"],
        "threshold_95": stats["dist_thresh_95"],
        "threshold_99": stats["dist_thresh_99"],
        "test_samples_evaluated": num_test,
        "high_reliability_count": high_cnt,
        "medium_reliability_count": med_cnt,
        "low_reliability_count": low_cnt,
        "in_distribution_count": in_dist_cnt,
        "possibly_ood_count": ood_cnt
    }

    with open(outputs_dir / "ood_eval_results.json", "w") as f:
        json.dump(ood_eval_results, f, indent=4)

    # Generate Markdown Report
    report_path = reports_dir / "ood_reliability_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 8 OOD & Forensic Reliability Report\n\n")
        f.write("**Evaluation Type:** In-Distribution vs Out-of-Distribution Calibration (No Retraining)\n")
        f.write(f"**Reference Population:** 2,414 Validation Samples (`val_features.pt` - Test Set Unused for Fitting)\n")
        f.write(f"**Evaluated Model:** 3V3 Champion Model (`experiments/05_full_3v3/models/best_model.pth`)\n\n")
        f.write("---\n\n")
        f.write("## 1. Reference Distribution Calibration (Validation Set)\n\n")
        f.write(f"- **Total Validation Samples Fitted:** {stats['val_samples']:,} (Real: {stats['val_real_count']}, Fake: {stats['val_fake_count']})\n")
        f.write(f"- **Feature Space:** 512-D Fused Multimodal Embedding Space\n")
        f.write(f"- **Distance Metric:** Standardized Class-Conditional Euclidean Distance (Diagonal Mahalanobis)\n")
        f.write(f"- **In-Distribution Distance (95th Percentile Threshold):** `{stats['dist_thresh_95']:.2f}`\n")
        f.write(f"- **In-Distribution Distance (99th Percentile Threshold):** `{stats['dist_thresh_99']:.2f}`\n\n")
        f.write("---\n\n")
        f.write("## 2. Benchmark Test Set Evaluation (604 Samples)\n\n")
        f.write("| Reliability Tier | Sample Count | Percentage | Operational Meaning |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        f.write(f"| **HIGH** | {high_cnt} | {high_cnt/num_test*100:.1f}% | High confidence, aligned multimodal evidence, in-distribution geometry |\n")
        f.write(f"| **MEDIUM** | {med_cnt} | {med_cnt/num_test*100:.1f}% | Minor stream divergence or moderate compression variance |\n")
        f.write(f"| **LOW** | {low_cnt} | {low_cnt/num_test*100:.1f}% | Extreme distance from learned manifold, small face, or screen capture |\n\n")
        f.write("---\n\n")
        f.write("## 3. Strict Non-Overriding Guarantee (Rule 7)\n\n")
        f.write("- The OOD and reliability layers **NEVER** alter the classifier verdict from Real to Fake or Fake to Real.\n")
        f.write("- OOD flags serve purely to advise human reviewers and the LLM agent on how much certainty should be attached to the detector output.\n")

    print(f"[+] Saved OOD report to: {report_path}\n")


if __name__ == "__main__":
    run_ood_evaluation()
