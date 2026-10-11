"""
tests/test_application.py

Comprehensive Automated Test Suite for Multimodal Deepfake Forensic Web Application.

Covers:
1. API health endpoint (/api/health)
2. Model singleton loading & weight verification
3. Valid video with audio (end-to-end inference)
4. Video without audio (vision-only fallback, audio bypass)
5. Silent audio handling
6. Corrupted audio handling
7. Corrupted video handling
8. Invalid file upload validation (rejected extension)
9. Agent evidence generation & reasoning grounding
10. Complete result JSON structure validation
11. Demo catalog & demo evaluation endpoint
"""

import os
import io
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2
import torch
import torchaudio

from fastapi.testclient import TestClient

from app.main import app
from app.config import (
    CHECKPOINT_PATH,
    CLASSIFIER_THRESHOLD,
    AV_CONSISTENCY_THRESHOLD,
    ALLOWED_EXTENSIONS,
)
from app.inference_service import InferenceService
from app.video_processor import VideoProcessor
from app.demo_service import DemoService
from agent.forensic_agent import ForensicAgent
from agent.reasoning import ForensicReasoningEngine


class TestApplicationSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.inference_service = InferenceService.get_instance()
        cls.inference_service.load_model()
        cls.video_processor = VideoProcessor()
        cls.demo_service = DemoService()
        cls.agent = ForensicAgent()

        # Create temporary synthetic test assets
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # 1. Create synthetic video file with OpenCV
        cls.valid_video_path = cls.temp_dir / "synthetic_valid_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(cls.valid_video_path), fourcc, 20.0, (224, 224))
        for _ in range(30):
            # Create a simple colored frame
            frame = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        # 2. Create synthetic audio file (sine wave)
        cls.valid_audio_path = cls.temp_dir / "synthetic_audio.wav"
        t = torch.linspace(0, 5, 5 * 16000)
        sine_wave = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        torchaudio.save(str(cls.valid_audio_path), sine_wave, 16000)

        # 3. Create silent audio file
        cls.silent_audio_path = cls.temp_dir / "silent_audio.wav"
        torchaudio.save(str(cls.silent_audio_path), torch.zeros(1, 16000), 16000)

        # 4. Create corrupted audio file
        cls.corrupted_audio_path = cls.temp_dir / "corrupted_audio.wav"
        with open(cls.corrupted_audio_path, "wb") as f:
            f.write(b"NOT_A_VALID_WAV_HEADER_CORRUPTED_BYTES" * 100)

        # 5. Create corrupted video file
        cls.corrupted_video_path = cls.temp_dir / "corrupted_video.mp4"
        with open(cls.corrupted_video_path, "wb") as f:
            f.write(b"CORRUPTED_VIDEO_DATA" * 50)

        # 6. Create invalid file type (.txt)
        cls.invalid_file_path = cls.temp_dir / "test_doc.txt"
        with open(cls.invalid_file_path, "w") as f:
            f.write("This is a text document, not a video.")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. API Health Endpoint
    # -------------------------------------------------------------
    def test_api_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("encoders", data)
        self.assertEqual(len(data["encoders"]["vision"]), 3)
        self.assertEqual(len(data["encoders"]["audio"]), 3)
        self.assertTrue(Path(data["checkpoint_path"]).exists())

    # -------------------------------------------------------------
    # 2. Model Singleton Loading
    # -------------------------------------------------------------
    def test_model_singleton_loading(self):
        instance1 = InferenceService.get_instance()
        instance2 = InferenceService.get_instance()
        self.assertIs(instance1, instance2)
        self.assertTrue(instance1.is_loaded)
        self.assertIsNotNone(instance1.model)
        
        # Verify model parameters are frozen (no gradients computed)
        for param in instance1.model.parameters():
            self.assertFalse(param.requires_grad)

    # -------------------------------------------------------------
    # 3. Valid Video with Audio Inference
    # -------------------------------------------------------------
    def test_valid_video_with_audio_inference(self):
        images = torch.randn(1, 16, 3, 224, 224)
        waveform = torch.randn(1, 160000)
        
        result = self.inference_service.run_inference(
            images_tensor=images,
            waveform_tensor=waveform,
            audio_available=True,
            video_filename="test_multimodal.mp4",
        )

        self.assertIn(result["prediction"], ["REAL", "FAKE"])
        self.assertGreaterEqual(result["confidence"], 0.50)
        self.assertTrue(result["audio_available"])
        self.assertIn("vision", result["fusion_weights"])
        self.assertIn("audio", result["fusion_weights"])
        self.assertAlmostEqual(
            result["fusion_weights"]["vision"] + result["fusion_weights"]["audio"],
            1.0,
            places=3,
        )
        self.assertIsNotNone(result["av_consistency"])
        self.assertIn("agent_analysis", result)

    # -------------------------------------------------------------
    # 4. Missing Audio Fallback (Audio Bypassed, Vision-Only)
    # -------------------------------------------------------------
    def test_missing_audio_fallback(self):
        images = torch.randn(1, 16, 3, 224, 224)
        zero_waveform = torch.zeros(1, 160000)

        result = self.inference_service.run_inference(
            images_tensor=images,
            waveform_tensor=zero_waveform,
            audio_available=False,
            video_filename="test_missing_audio.mp4",
        )

        self.assertFalse(result["audio_available"])
        self.assertEqual(result["fusion_weights"]["vision"], 1.0)
        self.assertEqual(result["fusion_weights"]["audio"], 0.0)
        self.assertIsNone(result["av_consistency"])
        self.assertTrue(np.isfinite(result["confidence"]))
        self.assertIn(result["prediction"], ["REAL", "FAKE"])
        
        # Verify Agent recognizes missing audio
        agent_res = result["agent_analysis"]
        self.assertIn("Vision Only", agent_res["fusion_breakdown"]["mode"])
        audio_evidence_str = " ".join(agent_res["audio_evidence"])
        self.assertIn("bypassed", audio_evidence_str.lower())

    # -------------------------------------------------------------
    # 5. Silent Audio Handling
    # -------------------------------------------------------------
    def test_silent_audio_handling(self):
        detector = self.video_processor.audio_detector
        is_valid, reason = detector.check_audio_file(self.silent_audio_path)
        self.assertFalse(is_valid)
        self.assertIn("silent", reason.lower())

    # -------------------------------------------------------------
    # 6. Corrupted Audio Handling
    # -------------------------------------------------------------
    def test_corrupted_audio_handling(self):
        detector = self.video_processor.audio_detector
        is_valid, reason = detector.check_audio_file(self.corrupted_audio_path)
        self.assertFalse(is_valid)
        self.assertIn("corrupted", reason.lower())

    # -------------------------------------------------------------
    # 7. Corrupted Video Handling
    # -------------------------------------------------------------
    def test_corrupted_video_upload(self):
        with open(self.corrupted_video_path, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                files={"video": ("corrupted_video.mp4", f, "video/mp4")},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("failed", response.json()["detail"].lower())

    # -------------------------------------------------------------
    # 8. Invalid File Upload Validation
    # -------------------------------------------------------------
    def test_invalid_file_upload(self):
        with open(self.invalid_file_path, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                files={"video": ("test_doc.txt", f, "text/plain")},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported file extension", response.json()["detail"].lower())

    # -------------------------------------------------------------
    # 9. Agent Evidence Generation & Reasoning Grounding
    # -------------------------------------------------------------
    def test_agent_evidence_generation(self):
        test_raw = {
            "prediction": "FAKE",
            "confidence": 0.992,
            "probability_real": 0.008,
            "probability_fake": 0.992,
            "audio_available": True,
            "fusion_weights": {"vision": 0.70, "audio": 0.30},
            "av_consistency": 0.052,
        }
        res = self.agent.process_evidence(test_raw, "synthetic_test.mp4")
        
        self.assertEqual(res["agent_analysis"]["verdict"], "FAKE")
        self.assertEqual(res["agent_analysis"]["risk_level"], "CRITICAL")
        self.assertIn("Quarantine media", res["agent_analysis"]["recommendation"])
        self.assertIn("cross_modal_analysis", res["agent_analysis"])
        self.assertTrue(res["agent_analysis"]["cross_modal_analysis"]["is_mismatched"])
        self.assertIn("# Forensic Deepfake Analysis Report", res["markdown_report"])

    # -------------------------------------------------------------
    # 10. Demo Catalog & Demo Evaluation Endpoints
    # -------------------------------------------------------------
    def test_demo_catalog_and_evaluation(self):
        # Catalog endpoint
        res_cat = self.client.get("/api/demo/samples")
        self.assertEqual(res_cat.status_code, 200)
        samples = res_cat.json()["samples"]
        self.assertGreaterEqual(len(samples), 4)

        # Evaluate demo case
        case_id = samples[0]["id"]
        res_demo = self.client.post(f"/api/demo/analyze/{case_id}")
        self.assertEqual(res_demo.status_code, 200)
        demo_data = res_demo.json()
        self.assertEqual(demo_data["demo_case_id"], case_id)
        self.assertIn("agent_analysis", demo_data)
        self.assertIn("fusion_weights", demo_data)

    # -------------------------------------------------------------
    # 11. Modality-Specific Verdicts & Technical Details Schema
    # -------------------------------------------------------------
    def test_modality_specific_verdicts(self):
        """
        Verify that inference outputs distinguish between:
        1. Visual authenticity (video_prediction)
        2. Audio authenticity (audio_status / audio_prediction)
        3. Audio-video consistency (audio_video_consistency)
        4. Overall classification (overall_prediction)
        """
        images = torch.randn(1, 16, 3, 224, 224)
        waveform = torch.randn(1, 160000)
        
        result = self.inference_service.run_inference(
            images_tensor=images,
            waveform_tensor=waveform,
            audio_available=True,
            video_filename="test_modality.mp4",
        )

        # 1. Overall
        self.assertIn("overall_prediction", result)
        self.assertIn(result["overall_prediction"], ["REAL", "FAKE"])
        self.assertIn("overall_confidence", result)
        self.assertIn("overall_probability", result)

        # 2. Video Analysis
        self.assertIn("video_prediction", result)
        self.assertIn(result["video_prediction"], ["REAL", "FAKE"])
        self.assertIn("video_confidence", result)
        self.assertIn("video_evidence", result)
        self.assertIsInstance(result["video_evidence"], list)

        # 3. Audio Analysis
        self.assertIn("audio_prediction", result)
        self.assertIn("audio_status", result)
        self.assertEqual(result["audio_status"], "EVALUATED_JOINTLY")
        self.assertIsNone(result["audio_probability"])  # Standalone head not trained
        self.assertIn("audio_evidence", result)

        # 4. Consistency
        self.assertIn("audio_video_consistency", result)
        self.assertIn(result["audio_video_consistency"]["status"], ["CONSISTENT", "MISMATCH_DETECTED", "INCONCLUSIVE"])
        self.assertEqual(result["audio_video_consistency"]["threshold"], 0.1177)

        # 5. Technical details
        self.assertIn("technical_details", result)
        self.assertIn("fusion_weights", result["technical_details"])
        self.assertIn("inference_time_ms", result["technical_details"])


if __name__ == "__main__":
    unittest.main()

