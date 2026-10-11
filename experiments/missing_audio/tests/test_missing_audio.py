"""
experiments/missing_audio/tests/test_missing_audio.py

Automated Test Suite for Missing Audio Detection and Adaptive Vision-Only Inference.

Tests 4 Scenarios:
1. Normal Video (Usable audio stream)
2. Silent Video (Zero amplitude waveform)
3. Missing Audio File (Non-existent path)
4. Corrupted Audio File (Invalid / garbage audio bytes)
"""

import os
import sys
import tempfile
import torch
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from preprocessing.audio_detector import AudioDetector, check_audio_availability
from models.multimodal_model import MultimodalDeepfakeDetector


class TestMissingAudioInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = MultimodalDeepfakeDetector().to(cls.device)
        cls.model.eval()

        # Dummy visual input: (B=1, T=16, C=3, H=224, W=224)
        cls.dummy_images = torch.randn(1, 16, 3, 224, 224).to(cls.device)

        # Temporary folder for corrupted / missing file tests
        cls.temp_dir = tempfile.TemporaryDirectory()

        # Create corrupted audio file
        cls.corrupt_audio_path = os.path.join(cls.temp_dir.name, "corrupt.wav")
        with open(cls.corrupt_audio_path, "wb") as f:
            f.write(b"GARBAGE_BYTES_INVALID_WAV_HEADER_1234567890")

        # Non-existent audio path
        cls.missing_audio_path = os.path.join(cls.temp_dir.name, "does_not_exist.wav")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_case_1_normal_audio(self):
        """Case 1: Normal Audio Stream (Usable audio present)"""
        detector = AudioDetector()
        valid_waveform = torch.randn(16000)

        is_available, reason = detector.check_waveform(valid_waveform)
        self.assertTrue(is_available, f"Expected True for normal audio, got {is_available} ({reason})")

        waveform_batch = torch.randn(1, 160000).to(self.device)
        with torch.no_grad():
            output = self.model(self.dummy_images, waveform_batch, audio_available=True)

        self.assertTrue(output["audio_available"])
        self.assertEqual(output["fusion_mode"], "Adaptive Multimodal")
        self.assertIsNotNone(output["audio_embedding"])
        self.assertEqual(output["logits"].shape, (1, 2))
        self.assertAlmostEqual(output["vision_weight"].item() + output["audio_weight"].item(), 1.0, places=4)

    def test_case_2_silent_audio(self):
        """Case 2: Silent Video (Waveform is all zeros)"""
        detector = AudioDetector()
        silent_waveform = torch.zeros(16000)

        is_available, reason = detector.check_waveform(silent_waveform)
        self.assertFalse(is_available, f"Expected False for silent audio, got {is_available}")

        waveform_batch = torch.zeros(1, 160000).to(self.device)
        with torch.no_grad():
            output = self.model(self.dummy_images, waveform_batch)

        self.assertFalse(output["audio_available"])
        self.assertEqual(output["fusion_mode"], "Vision Only")
        self.assertIsNone(output["audio_embedding"])
        self.assertEqual(output["logits"].shape, (1, 2))
        self.assertEqual(output["vision_weight"].item(), 1.0)
        self.assertEqual(output["audio_weight"].item(), 0.0)

    def test_case_3_missing_audio_file(self):
        """Case 3: Missing Audio File (File does not exist)"""
        detector = AudioDetector()
        is_available, reason = detector.check_audio_file(self.missing_audio_path)
        self.assertFalse(is_available)
        self.assertIn("does not exist", reason)

        with torch.no_grad():
            output = self.model(self.dummy_images, waveform=None, audio_available=False)

        self.assertFalse(output["audio_available"])
        self.assertEqual(output["fusion_mode"], "Vision Only")
        self.assertIsNone(output["audio_embedding"])
        self.assertEqual(output["logits"].shape, (1, 2))
        self.assertEqual(output["vision_weight"].item(), 1.0)
        self.assertEqual(output["audio_weight"].item(), 0.0)

    def test_case_4_corrupted_audio_file(self):
        """Case 4: Corrupted Audio File (Invalid binary bytes)"""
        detector = AudioDetector()
        is_available, reason = detector.check_audio_file(self.corrupt_audio_path)
        self.assertFalse(is_available)
        self.assertIn("failure", reason.lower())

        # Simulate fallback to vision-only inference
        with torch.no_grad():
            output = self.model(self.dummy_images, waveform=None, audio_available=is_available)

        self.assertFalse(output["audio_available"])
        self.assertEqual(output["fusion_mode"], "Vision Only")
        self.assertEqual(output["logits"].shape, (1, 2))
        self.assertEqual(output["vision_weight"].item(), 1.0)
        self.assertEqual(output["audio_weight"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
