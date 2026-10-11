"""
experiments/03_vision_swin/tests/test_vision_swin.py

Phase-3 Verification Test Suite for Video Swin Visual Feature Extraction Branch.

Tests:
1. Video Swin Forward Test
2. EfficientNet Forward Test
3. Vision Aggregation Forward Test
4. Complete Model Forward Test
5. Missing Audio Test
6. Vision-only Mode Test
7. Checkpoint Compatibility Test
8. Factory Registration Test
9. Feature Export Test
10. PCA/t-SNE Generation Test
"""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn as nn

from models.base_vision_encoder import BaseVisionEncoder
from models.efficientnet import EfficientNetEncoder
from models.video_swin import VideoSwinEncoder
from models.vision_encoder_factory import VisionEncoderFactory
from models.vision_feature_aggregation import VisionFeatureAggregation
from models.multimodal_model import MultimodalDeepfakeDetector


class TestVideoSwinFramework(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.batch_size = 2
        cls.num_frames = 16
        cls.dummy_images = torch.randn(cls.batch_size, cls.num_frames, 3, 224, 224, device=cls.device)
        cls.dummy_waveform = torch.randn(cls.batch_size, 160000, device=cls.device)

    def test_01_video_swin_forward(self):
        """Test VideoSwinEncoder output shape (B, 768)."""
        encoder = VideoSwinEncoder(pretrained=False).to(self.device)
        with torch.no_grad():
            out = encoder(self.dummy_images)
        self.assertEqual(out.shape, (self.batch_size, 768))
        self.assertEqual(encoder.output_dim, 768)

    def test_02_efficientnet_forward(self):
        """Test EfficientNetEncoder output shape (B, 1280)."""
        encoder = EfficientNetEncoder(pretrained=False).to(self.device)
        with torch.no_grad():
            out = encoder(self.dummy_images)
        self.assertEqual(out.shape, (self.batch_size, 1280))
        self.assertEqual(encoder.output_dim, 1280)

    def test_03_vision_aggregation_forward(self):
        """Test VisionFeatureAggregation (Concat 2048 -> Single Linear 1280)."""
        eff_features = torch.randn(self.batch_size, 1280, device=self.device)
        swin_features = torch.randn(self.batch_size, 768, device=self.device)

        aggregator = VisionFeatureAggregation(efficientnet_dim=1280, video_swin_dim=768, output_dim=1280).to(self.device)
        with torch.no_grad():
            fused = aggregator(eff_features, swin_features)

        self.assertEqual(fused.shape, (self.batch_size, 1280))

    def test_04_complete_model_forward(self):
        """Test full MultimodalDeepfakeDetector dictionary output contract."""
        model = MultimodalDeepfakeDetector(enable_efficientnet=True, enable_video_swin=True).to(self.device)
        with torch.no_grad():
            out = model(self.dummy_images, self.dummy_waveform, return_intermediate_features=True)

        expected_keys = [
            "logits", "vision_embedding", "audio_embedding", "fused_embedding",
            "attention", "contrastive_loss", "similarity", "audio_available",
            "vision_weight", "audio_weight", "fusion_mode",
            "efficientnet_features", "video_swin_features", "fused_vision_features"
        ]
        for key in expected_keys:
            self.assertIn(key, out, f"Missing expected key: {key}")

        self.assertEqual(out["logits"].shape, (self.batch_size, 2))
        self.assertEqual(out["vision_embedding"].shape, (self.batch_size, 256))
        self.assertEqual(out["audio_embedding"].shape, (self.batch_size, 256))
        self.assertEqual(out["fused_embedding"].shape, (self.batch_size, 512))

    def test_05_missing_audio(self):
        """Test model execution when audio waveform is missing or zero."""
        model = MultimodalDeepfakeDetector().to(self.device)
        # Test waveform=None
        with torch.no_grad():
            out_none = model(self.dummy_images, waveform=None)
        self.assertFalse(out_none["audio_available"])
        self.assertEqual(out_none["fusion_mode"], "Vision Only")
        self.assertTrue(torch.allclose(out_none["vision_weight"], torch.ones(self.batch_size, device=self.device)))
        self.assertTrue(torch.allclose(out_none["audio_weight"], torch.zeros(self.batch_size, device=self.device)))

        # Test silent audio (zeros)
        silent_waveform = torch.zeros(self.batch_size, 160000, device=self.device)
        with torch.no_grad():
            out_silent = model(self.dummy_images, waveform=silent_waveform)
        self.assertFalse(out_silent["audio_available"])

    def test_06_vision_only_mode(self):
        """Test explicit audio_available=False vision-only mode."""
        model = MultimodalDeepfakeDetector().to(self.device)
        with torch.no_grad():
            out = model(self.dummy_images, self.dummy_waveform, audio_available=False)
        self.assertFalse(out["audio_available"])
        self.assertIsNone(out["audio_embedding"])

    def test_07_checkpoint_compatibility(self):
        """Test saving and loading model state_dict backward compatibility."""
        model = MultimodalDeepfakeDetector().to(self.device)
        model.eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "model.pth"
            torch.save(model.state_dict(), tmp_path)
            model_loaded = MultimodalDeepfakeDetector().to(self.device)
            model_loaded.load_state_dict(torch.load(tmp_path))
            model_loaded.eval()

            with torch.no_grad():
                out_orig = model(self.dummy_images, self.dummy_waveform)
                out_loaded = model_loaded(self.dummy_images, self.dummy_waveform)

            self.assertTrue(torch.allclose(out_orig["logits"], out_loaded["logits"], atol=1e-5))

    def test_08_factory_registration(self):
        """Test VisionEncoderFactory registration and retrieval."""
        registered = VisionEncoderFactory.list_registered()
        self.assertIn("efficientnet", registered)
        self.assertIn("video_swin", registered)

        eff_encoder = VisionEncoderFactory.create("efficientnet", pretrained=False)
        self.assertIsInstance(eff_encoder, BaseVisionEncoder)
        self.assertEqual(eff_encoder.output_dim, 1280)

        swin_encoder = VisionEncoderFactory.create("video_swin", pretrained=False)
        self.assertIsInstance(swin_encoder, BaseVisionEncoder)
        self.assertEqual(swin_encoder.output_dim, 768)

    def test_09_feature_export(self):
        """Test intermediate visual feature export formatting."""
        model = MultimodalDeepfakeDetector().to(self.device)
        with torch.no_grad():
            out = model(self.dummy_images, self.dummy_waveform, return_intermediate_features=True)

        eff_np = out["efficientnet_features"].cpu().numpy()
        swin_np = out["video_swin_features"].cpu().numpy()
        fused_np = out["fused_vision_features"].cpu().numpy()

        self.assertEqual(eff_np.shape, (self.batch_size, 1280))
        self.assertEqual(swin_np.shape, (self.batch_size, 768))
        self.assertEqual(fused_np.shape, (self.batch_size, 1280))

    def test_10_pca_tsne_generation(self):
        """Test PCA and t-SNE projection execution on synthetic feature arrays."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        features = np.random.randn(10, 1280)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        self.assertEqual(pca_result.shape, (10, 2))

        tsne = TSNE(n_components=2, perplexity=3, random_state=42)
        tsne_result = tsne.fit_transform(features)
        self.assertEqual(tsne_result.shape, (10, 2))


    def test_11_partial_freezing(self):
        """Test partial freezing strategy on visual encoders."""
        eff_partial = EfficientNetEncoder(pretrained=False, freeze_strategy="partial")
        eff_frozen = [p.requires_grad for p in list(eff_partial.features.children())[0].parameters()]
        eff_trainable = [p.requires_grad for p in list(eff_partial.features.children())[-1].parameters()]
        self.assertTrue(all(not grad for grad in eff_frozen))
        self.assertTrue(all(grad for grad in eff_trainable))

        swin_partial = VideoSwinEncoder(pretrained=False, freeze_strategy="partial")
        swin_frozen = [p.requires_grad for p in list(swin_partial.backbone.features.children())[0].parameters()]
        swin_trainable = [p.requires_grad for p in list(swin_partial.backbone.features.children())[-1].parameters()]
        self.assertTrue(all(not grad for grad in swin_frozen))
        self.assertTrue(all(grad for grad in swin_trainable))


if __name__ == "__main__":
    unittest.main()
