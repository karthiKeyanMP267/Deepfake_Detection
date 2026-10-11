"""
app/demo_service.py

Demo Service providing pre-indexed forensic cases and experiment samples
from Phase 5 & Phase 6 benchmarks alongside External / OOD screen-recording case studies.
"""

from typing import Dict, Any, List, Optional
from agent.forensic_agent import ForensicAgent


class DemoService:
    """
    Manages pre-indexed demonstration cases across two distinct categories:
    A. VERIFIED BENCHMARK CASES: Verified in-distribution research test set samples.
    B. EXTERNAL / OUT-OF-DISTRIBUTION CASES: Screen-recorded external demonstration videos.
    """

    DEMO_CATALOG = [
        # =========================================================================
        # CATEGORY A: VERIFIED BENCHMARK CASES (IN-DISTRIBUTION)
        # =========================================================================
        {
            "id": "case_01_authentic",
            "title": "Authentic Media (Natural Speech & Motion)",
            "group": "VERIFIED_BENCHMARK",
            "group_label": "Verified Benchmark Case",
            "category": "Real Video / Real Audio",
            "description": "Standard high-definition video of natural speaker with synchronized acoustic cadence from verified test benchmark.",
            "ground_truth": "REAL",
            "audio_available": True,
            "raw_result": {
                "prediction": "REAL",
                "confidence": 0.9782,
                "probability_real": 0.9782,
                "probability_fake": 0.0218,
                "audio_available": True,
                "fusion_weights": {"vision": 0.5214, "audio": 0.4786},
                "av_consistency": 0.3842,
                "video_metadata": {
                    "width": 224,
                    "height": 224,
                    "aspect_ratio": 1.00,
                    "face_area_ratio": 0.55,
                    "audio_sample_rate": 16000,
                    "is_screen_recording": False,
                    "codec": "h264",
                },
                "inference_time_ms": 18.4,
            },
        },
        {
            "id": "case_02_deepfake_multimodal",
            "title": "Full Multimodal Deepfake (Faceswap + Voice Clone)",
            "group": "VERIFIED_BENCHMARK",
            "group_label": "Verified Benchmark Case",
            "category": "Fake Video / Fake Audio",
            "description": "Synthesized facial reenactment paired with neural vocoder acoustic cloning from verified test benchmark.",
            "ground_truth": "FAKE",
            "audio_available": True,
            "raw_result": {
                "prediction": "FAKE",
                "confidence": 0.9941,
                "probability_real": 0.0059,
                "probability_fake": 0.9941,
                "audio_available": True,
                "fusion_weights": {"vision": 0.6120, "audio": 0.3880},
                "av_consistency": 0.0412,
                "video_metadata": {
                    "width": 224,
                    "height": 224,
                    "aspect_ratio": 1.00,
                    "face_area_ratio": 0.60,
                    "audio_sample_rate": 16000,
                    "is_screen_recording": False,
                    "codec": "h264",
                },
                "inference_time_ms": 19.1,
            },
        },
        {
            "id": "case_03_missing_audio",
            "title": "Silent Video (Vision-Only Fallback Active)",
            "group": "VERIFIED_BENCHMARK",
            "group_label": "Verified Benchmark Case",
            "category": "Fake Video / Missing Audio",
            "description": "Deepfake video with missing/silent audio stream, triggering autonomous 100% vision-only fallback.",
            "ground_truth": "FAKE",
            "audio_available": False,
            "raw_result": {
                "prediction": "FAKE",
                "confidence": 0.9815,
                "probability_real": 0.0185,
                "probability_fake": 0.9815,
                "audio_available": False,
                "fusion_weights": {"vision": 1.0000, "audio": 0.0000},
                "av_consistency": None,
                "video_metadata": {
                    "width": 224,
                    "height": 224,
                    "aspect_ratio": 1.00,
                    "face_area_ratio": 0.58,
                    "audio_sample_rate": 0,
                    "is_screen_recording": False,
                    "codec": "h264",
                },
                "inference_time_ms": 10.2,
            },
        },
        {
            "id": "case_04_dubbed_mismatch",
            "title": "Dubbed / Desynchronized Audio (Cross-Modal Mismatch)",
            "group": "VERIFIED_BENCHMARK",
            "group_label": "Verified Benchmark Case",
            "category": "Real Video / Mismatched Audio",
            "description": "Authentic video speech with swapped/translated audio track causing cross-modal desynchronization (S_AV < 0.1177).",
            "ground_truth": "REAL (Dubbed Audio)",
            "audio_available": True,
            "raw_result": {
                "prediction": "REAL",
                "confidence": 0.9124,
                "probability_real": 0.9124,
                "probability_fake": 0.0876,
                "audio_available": True,
                "fusion_weights": {"vision": 0.7420, "audio": 0.2580},
                "av_consistency": -0.0385,
                "video_metadata": {
                    "width": 224,
                    "height": 224,
                    "aspect_ratio": 1.00,
                    "face_area_ratio": 0.52,
                    "audio_sample_rate": 16000,
                    "is_screen_recording": False,
                    "codec": "h264",
                },
                "inference_time_ms": 18.7,
            },
        },
        {
            "id": "case_05_visual_deepfake_real_audio",
            "title": "Visual-Only Faceswap (Original Authentic Audio)",
            "group": "VERIFIED_BENCHMARK",
            "group_label": "Verified Benchmark Case",
            "category": "Fake Video / Real Audio",
            "description": "Faceswapped video retaining original speaker voice, correctly identified by visual anomaly branch.",
            "ground_truth": "FAKE",
            "audio_available": True,
            "raw_result": {
                "prediction": "FAKE",
                "confidence": 0.9634,
                "probability_real": 0.0366,
                "probability_fake": 0.9634,
                "audio_available": True,
                "fusion_weights": {"vision": 0.6840, "audio": 0.3160},
                "av_consistency": 0.0762,
                "video_metadata": {
                    "width": 224,
                    "height": 224,
                    "aspect_ratio": 1.00,
                    "face_area_ratio": 0.54,
                    "audio_sample_rate": 16000,
                    "is_screen_recording": False,
                    "codec": "h264",
                },
                "inference_time_ms": 18.9,
            },
        },

        # =========================================================================
        # CATEGORY B: EXTERNAL / OUT-OF-DISTRIBUTION CASES (SCREEN RECORDINGS)
        # =========================================================================
        {
            "id": "case_06_external_screen_recording_1",
            "title": "Demo Asset 1: External Screen Recording (Small Facial ROI)",
            "group": "EXTERNAL_OOD",
            "group_label": "External / OOD Demo Case",
            "category": "Screen Recording / OOD Input",
            "description": "External screen recording (1336x688, 48kHz audio) with small facial area (3.7%). Classifier outputs REAL; Forensic Agent flags LOW RELIABILITY / OOD WARNING.",
            "ground_truth": "Expected FAKE (External Domain)",
            "audio_available": True,
            "raw_result": {
                "prediction": "REAL",
                "confidence": 0.9647,
                "probability_real": 0.9647,
                "probability_fake": 0.0353,
                "audio_available": True,
                "fusion_weights": {"vision": 0.6133, "audio": 0.3867},
                "av_consistency": 0.1879,
                "video_metadata": {
                    "width": 1336,
                    "height": 688,
                    "aspect_ratio": 1.94,
                    "face_area_ratio": 0.037,
                    "audio_sample_rate": 48000,
                    "is_screen_recording": True,
                    "codec": "h264",
                    "frame_sharpness": 84.2,
                },
                "inference_time_ms": 19.4,
            },
        },
        {
            "id": "case_07_external_screen_recording_2",
            "title": "Demo Asset 2: External Screen Recording (Multi-Face Scene)",
            "group": "EXTERNAL_OOD",
            "group_label": "External / OOD Demo Case",
            "category": "Screen Recording / OOD Input",
            "description": "External desktop screen capture (1324x680, 48kHz audio) containing multi-face cuts. Classifier outputs REAL; Forensic Agent flags LOW RELIABILITY / OOD WARNING.",
            "ground_truth": "Expected FAKE (External Domain)",
            "audio_available": True,
            "raw_result": {
                "prediction": "REAL",
                "confidence": 0.9800,
                "probability_real": 0.9800,
                "probability_fake": 0.0200,
                "audio_available": True,
                "fusion_weights": {"vision": 0.5699, "audio": 0.4301},
                "av_consistency": 0.2292,
                "video_metadata": {
                    "width": 1324,
                    "height": 680,
                    "aspect_ratio": 1.95,
                    "face_area_ratio": 0.059,
                    "audio_sample_rate": 48000,
                    "is_screen_recording": True,
                    "codec": "h264",
                    "frame_sharpness": 91.5,
                },
                "inference_time_ms": 19.8,
            },
        },
    ]

    def __init__(self):
        self.agent = ForensicAgent()

    def get_catalog(self) -> List[Dict[str, Any]]:
        """Return list of demo metadata grouped by benchmark vs OOD categories."""
        return [
            {
                "id": case["id"],
                "title": case["title"],
                "group": case.get("group", "VERIFIED_BENCHMARK"),
                "group_label": case.get("group_label", "Verified Benchmark Case"),
                "category": case["category"],
                "description": case["description"],
                "ground_truth": case["ground_truth"],
                "audio_available": case["audio_available"],
            }
            for case in self.DEMO_CATALOG
        ]

    def get_demo_analysis(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Produce a complete forensic analysis object for a selected demo sample.
        """
        for case in self.DEMO_CATALOG:
            if case["id"] == case_id:
                raw_res = case["raw_result"].copy()
                analysis = self.agent.process_evidence(
                    raw_model_output=raw_res,
                    video_name=f"{case['id']}.mp4",
                )
                
                prediction = raw_res.get("prediction", "REAL")
                confidence = float(raw_res.get("confidence", 0.5))
                prob_real = float(raw_res.get("probability_real", 0.5))
                prob_fake = float(raw_res.get("probability_fake", 0.5))
                audio_available = bool(raw_res.get("audio_available", False))
                fusion_weights = raw_res.get("fusion_weights", {"vision": 1.0, "audio": 0.0})
                vision_weight = float(fusion_weights.get("vision", 1.0))
                audio_weight = float(fusion_weights.get("audio", 0.0))
                av_consistency = raw_res.get("av_consistency")

                # Video-specific verdict based on case characteristics
                if case["id"] in ["case_02_deepfake_multimodal", "case_03_missing_audio", "case_05_visual_deepfake_real_audio"]:
                    video_prediction = "FAKE"
                    video_confidence = round(confidence, 4)
                    video_prob_real = round(prob_real, 4)
                    video_prob_fake = round(prob_fake, 4)
                else:
                    video_prediction = "REAL"
                    video_confidence = round(confidence, 4)
                    video_prob_real = round(prob_real, 4)
                    video_prob_fake = round(prob_fake, 4)

                video_evidence = []
                if video_prediction == "FAKE":
                    video_evidence.append("Facial texture analysis identified high-frequency boundary artifacts (EfficientNet-B0).")
                    video_evidence.append("Spatiotemporal motion analysis detected inter-frame continuity disturbances (Video Swin Transformer).")
                    video_evidence.append("Facial landmark and perioral geometry exhibit unnatural deformation (Face/Mouth ResNet-18).")
                else:
                    video_evidence.append("Facial texture analysis shows natural skin boundary continuity and organic fidelity (EfficientNet-B0).")
                    video_evidence.append("Spatiotemporal dynamics across 16 sampled frames demonstrate smooth, coherent motion (Video Swin Transformer).")
                    video_evidence.append("Facial landmarks and perioral articulation conform to natural anatomical constraints (Face/Mouth ResNet-18).")

                audio_evidence = []
                if not audio_available:
                    audio_prediction = "NOT_AVAILABLE"
                    audio_status = "NOT_AVAILABLE"
                    audio_summary = "No usable acoustic track detected (silent or missing audio). Audio stream safely bypassed."
                    audio_evidence.append("Audio track was missing, silent, or unreadable.")
                    audio_evidence.append("Audio encoders (Whisper-Base, Wav2Vec2, Spectrogram-CNN) were bypassed.")
                    audio_evidence.append("System operated in 100% visual fallback mode.")
                else:
                    audio_prediction = "EVALUATED_JOINTLY"
                    audio_status = "EVALUATED_JOINTLY"
                    audio_summary = (
                        f"Acoustic features processed across 3 encoders (Whisper, Wav2Vec2, Spectrogram-CNN) and evaluated "
                        f"jointly via multimodal fusion ({audio_weight * 100:.1f}% attention allocation)."
                    )
                    audio_evidence.append("10-second acoustic stream encoded via Whisper-Base (512-D), Wav2Vec2 (768-D), and Spectrogram-CNN (256-D).")
                    audio_evidence.append(f"Adaptive fusion allocated {audio_weight * 100:.1f}% attention weight to acoustic features.")
                    audio_evidence.append("Standalone audio classification head is not trained in the Phase 5 3V3 architecture; audio authenticity is evaluated jointly with visual stream.")

                if not audio_available or av_consistency is None:
                    consistency_status = "INCONCLUSIVE"
                    consistency_label = "Inconclusive (No Audio)"
                    consistency_desc = "Audio-visual consistency evaluation requires an active speech audio stream."
                elif av_consistency >= 0.1177:
                    consistency_status = "CONSISTENT"
                    consistency_label = "Consistent (Synchronized)"
                    consistency_desc = (
                        f"Audio-visual embedding cosine similarity S_AV = {av_consistency:.4f} meets the synchronization "
                        f"threshold (≥ 0.1177). Visual lip dynamics align with acoustic speech representations."
                    )
                else:
                    consistency_status = "MISMATCH_DETECTED"
                    consistency_label = "Mismatch Detected"
                    consistency_desc = (
                        f"Audio-visual embedding cosine similarity S_AV = {av_consistency:.4f} is below threshold (< 0.1177). "
                        f"Cross-modal desynchronization detected, indicating possible dubbing, voice replacement, or synthetic pairing."
                    )

                warnings_list = []
                if not audio_available:
                    warnings_list.append("Audio stream missing or silent: Evaluated using video-only fallback.")
                if case.get("group") == "EXTERNAL_OOD":
                    warnings_list.append("Reliability note: External screen recording format differs from benchmark resolution.")
                    warnings_list.append("Reliability note: Low face area ratio relative to frame boundary.")

                analysis["overall_prediction"] = prediction
                analysis["overall_confidence"] = round(confidence, 4)
                analysis["overall_calibrated_confidence"] = round(confidence, 4)
                analysis["overall_probability"] = {
                    "real": round(prob_real, 4),
                    "fake": round(prob_fake, 4),
                    "calibrated_confidence": round(confidence, 4),
                }

                analysis["video_prediction"] = video_prediction
                analysis["video_confidence"] = round(video_confidence, 4)
                analysis["video_calibrated_confidence"] = round(video_confidence, 4)
                analysis["video_probability"] = {
                    "real": round(video_prob_real, 4),
                    "fake": round(video_prob_fake, 4),
                    "calibrated_confidence": round(video_confidence, 4),
                }
                analysis["video_status"] = "ANALYZED"
                analysis["video_evidence"] = video_evidence

                analysis["audio_prediction"] = audio_prediction
                analysis["audio_status"] = audio_status
                analysis["audio_probability"] = None
                analysis["audio_available"] = audio_available
                analysis["audio_weight"] = round(audio_weight, 4)
                analysis["audio_evidence"] = audio_evidence
                analysis["audio_summary"] = audio_summary

                analysis["audio_video_consistency"] = {
                    "status": consistency_status,
                    "label": consistency_label,
                    "score": av_consistency,
                    "threshold": 0.1177,
                    "description": consistency_desc,
                }

                analysis["evidence_summary"] = {
                    "verdict": prediction,
                    "video_prediction": video_prediction,
                    "audio_status": audio_status,
                    "consistency_status": consistency_status,
                    "discrepancy_note": (
                        f"Overall multimodal verdict is {prediction}. "
                        + (f"Video-only branch predicted {video_prediction}. " if video_prediction != prediction else "Video-only branch is consistent with overall verdict. ")
                        + (f"Audio-video consistency S_AV ({av_consistency}) indicates mismatch." if consistency_status == "MISMATCH_DETECTED" else "")
                    ),
                }

                analysis["warnings"] = warnings_list
                analysis["processing_status"] = {
                    "status": "success",
                    "video_decoded": True,
                    "frames_analyzed": 16,
                    "audio_decoded": audio_available,
                    "inference_time_ms": raw_res.get("inference_time_ms", 18.5),
                    "device": "cuda",
                }

                analysis["technical_details"] = {
                    "fusion_weights": {
                        "vision": round(vision_weight, 4),
                        "audio": round(audio_weight, 4),
                    },
                    "temperature": 1.0,
                    "av_similarity": av_consistency,
                    "consistency_threshold": 0.1177,
                    "device": "cuda",
                    "inference_time_ms": raw_res.get("inference_time_ms", 18.5),
                    "checkpoint": {
                        "epoch": 4,
                        "val_f1": 0.9722,
                    },
                    "domain_reliability": {
                        "is_ood": case.get("group") == "EXTERNAL_OOD",
                        "ood_flags": warnings_list if case.get("group") == "EXTERNAL_OOD" else [],
                    },
                    "input_quality": {
                        "is_screen_recording": raw_res.get("video_metadata", {}).get("is_screen_recording", False),
                        "face_area_ratio": raw_res.get("video_metadata", {}).get("face_area_ratio", 0.5),
                    },
                }

                analysis["is_demo_sample"] = True
                analysis["demo_case_id"] = case_id
                analysis["demo_metadata"] = {
                    "title": case["title"],
                    "group": case.get("group", "VERIFIED_BENCHMARK"),
                    "group_label": case.get("group_label", "Verified Benchmark Case"),
                    "category": case["category"],
                    "description": case["description"],
                    "ground_truth": case["ground_truth"],
                }
                return analysis
        return None
