"""
agent/report_generator.py

Forensic Report Generator for Multimodal Deepfake Detection.
Generates comprehensive, publication-ready Markdown and structured dictionary reports
with evidence grounding and out-of-distribution reliability assessments.
"""

from typing import Dict, Any
from datetime import datetime


class ForensicReportGenerator:
    """
    Formats multimodal forensic analysis results into structured markdown reports.
    """

    @classmethod
    def generate_markdown_report(
        cls,
        analysis_result: Dict[str, Any],
        video_filename: str = "uploaded_video.mp4",
    ) -> str:
        """
        Generate a comprehensive, structured forensic evaluation report in Markdown.
        """
        agent_analysis = analysis_result.get("agent_analysis", {})
        verdict = agent_analysis.get("verdict", analysis_result.get("prediction", "UNKNOWN"))
        confidence_pct = agent_analysis.get("confidence_percentage", analysis_result.get("confidence", 0.0) * 100.0)
        prob_real = agent_analysis.get("probability_real", analysis_result.get("probability_real", 0.0))
        prob_fake = agent_analysis.get("probability_fake", analysis_result.get("probability_fake", 0.0))
        risk_level = agent_analysis.get("risk_level", "UNKNOWN")
        recommendation = agent_analysis.get("recommendation", "N/A")
        narrative = agent_analysis.get("narrative_reasoning", "")

        fusion = analysis_result.get("fusion_weights", {})
        vision_w = fusion.get("vision", 1.0)
        audio_w = fusion.get("audio", 0.0)
        audio_available = analysis_result.get("audio_available", False)

        av_consistency = analysis_result.get("av_consistency")
        cross_modal = agent_analysis.get("cross_modal_analysis", {})
        consistency_status = cross_modal.get("status", "N/A")
        consistency_score_str = f"{av_consistency:.4f}" if av_consistency is not None else "N/A (Missing Audio)"

        domain_eval = agent_analysis.get("domain_reliability", {})
        forensic_status = agent_analysis.get("forensic_status", "HIGH RELIABILITY — IN-DISTRIBUTION VERIFIED")
        reliability_tag = agent_analysis.get("forensic_reliability", "HIGH")
        criteria = domain_eval.get("criteria_breakdown", {})
        ood_warning = domain_eval.get("ood_warning", False)
        ood_flags = domain_eval.get("ood_flags", [])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Forensic Deepfake Analysis Report

**Target Asset:** `{video_filename}`  
**Analysis Timestamp:** `{timestamp}`  
**Architecture:** Full 3-Vision + 3-Audio Multimodal (6 Encoders)  
**Evaluation Mode:** Inference-Only (`torch.inference_mode()`)  

---

## 1. Executive Summary

| Attribute | Forensic Value |
| :--- | :--- |
| **AI Classifier Verdict** | **{verdict}** |
| **Classifier Confidence Score** | **{confidence_pct:.2f}%** |
| **Probability (Real)** | {prob_real:.4f} |
| **Probability (Fake / Synthetic)** | {prob_fake:.4f} |
| **Forensic Reliability Status** | **{forensic_status}** |
| **Forensic Reliability Tier** | **{reliability_tag}** |
| **Assessed Risk Level** | **{risk_level}** |
| **Audio Availability** | {"YES (Multi-stream Active)" if audio_available else "NO (Vision-Only Fallback Active)"} |
| **Cross-Modal Consistency ($S_{{AV}}$)** | {consistency_score_str} ({consistency_status}) |

---

## 2. Forensic Reliability & Observable Domain Characteristics

> **Forensic Assessment Distinction:**  
> The system strictly distinguishes between *Classifier Verdict* ("{verdict}") and *Forensic Reliability* ("{reliability_tag}").
"""

        if ood_warning:
            report += f"""
> [!WARNING]
> **OUT-OF-DISTRIBUTION INPUT NOTICE:**  
> The classifier predicts **{verdict}**, however the forensic reliability layer identifies characteristics outside the verified benchmark distribution (`FakeAVCeleb`). The classification should therefore be interpreted cautiously.

### Out-of-Distribution Indicators:
"""
            for flag in ood_flags:
                report += f"- {flag}\n"
        else:
            report += """
> [!NOTE]
> **IN-DISTRIBUTION VERIFICATION:**  
> Input characteristics closely match the verified multimodal training and evaluation distribution.
"""

        report += f"""
### Observable Evidence Matrix:
| Parameter | Observed Characteristic | Benchmark Standard |
| :--- | :--- | :--- |
| **Input Spatial Resolution** | `{criteria.get('input_resolution', 'N/A')}` | Portrait / Square ($224 \\times 224$) |
| **Aspect Ratio** | `{criteria.get('aspect_ratio', 'N/A')}` | $1.00$ ($0.80 - 1.33$) |
| **Compression & Codec** | `{criteria.get('compression_codec', 'N/A')}` | Camera Direct Encode |
| **Frame Sharpness / Noise** | `{criteria.get('frame_sharpness', 'N/A')}` | High Uncompressed Detail |
| **Face / ROI Coverage** | `{criteria.get('face_roi_coverage', 'N/A')}` | > 25.0% of frame area |
| **Audio Channel Status** | `{criteria.get('audio_availability', 'N/A')}` | Synchronized Audio Track |
| **Audio Sample Rate** | `{criteria.get('audio_sample_rate', 'N/A')}` | $16,000$ Hz Mono |
| **Audio-Visual Consistency ($S_{{AV}}$)** | `{criteria.get('audio_visual_consistency', 'N/A')}` | >= 0.1177 (Aligned) |
| **Vision Modality Weight ($\\alpha_v$)** | `{criteria.get('vision_contribution', 'N/A')}` | Dynamic (50% - 75%) |
| **Audio Modality Weight ($\\alpha_a$)** | `{criteria.get('audio_contribution', 'N/A')}` | Dynamic (25% - 50%) |
| **Classifier Confidence** | `{criteria.get('classifier_confidence', 'N/A')}` | Calibrated Softmax |
| **Benchmark Alignment** | `{criteria.get('benchmark_alignment', 'N/A')}` | `FakeAVCeleb` Verified |

---

## 3. Modality Contribution & Adaptive Fusion

The adaptive fusion mechanism weighted the complementary sensory branches as follows:

- **Visual Modality Weight ($\\alpha_{{vision}}$):** `{vision_w:.4f}` ({vision_w * 100:.1f}%)
  - *Encoders:* EfficientNet-B0 (Spatial), Video Swin-Tiny (Temporal), Face/Mouth ResNet-18 (Articulation)
- **Audio Modality Weight ($\\alpha_{{audio}}$):** `{audio_w:.4f}` ({audio_w * 100:.1f}%)
  - *Encoders:* Whisper (Semantic), Wav2Vec2 (Acoustic), Spectrogram CNN (Time-Frequency)
- **Fusion Mode:** `{"Adaptive Multimodal (3V3)" if audio_available else "Vision-Only Fallback"}`

---

## 4. Forensic Evidence Breakdown

### Visual Branch Evidence:
"""
        for cue in agent_analysis.get("visual_evidence", []):
            report += f"- {cue}\n"

        report += "\n### Audio Branch Evidence:\n"
        for cue in agent_analysis.get("audio_evidence", []):
            report += f"- {cue}\n"

        report += f"""
### Cross-Modal Consistency & Synchronization:
- **Consistency Score ($S_{{AV}}$):** `{consistency_score_str}`
- **Threshold:** `0.1177` (Phase-6 Forensic Baseline)
- **Diagnostic:** {cross_modal.get("detail", "N/A")}

---

## 5. Agentic Reasoning & Synthesis

> {narrative}

---

## 6. Risk Assessment & Operational Recommendation

- **Assessed Risk Tier:** **{risk_level}**
- **Actionable Recommendation:** {recommendation}

---
*Report automatically generated by Agentic Multimodal Deepfake Forensic Analyzer.*
"""
        return report
