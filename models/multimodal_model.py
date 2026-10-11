"""
models/multimodal_model.py

Complete Multimodal Deepfake Detection Model supporting up to 6 Encoders (3 Vision + 3 Audio):

Vision Branch (3 Models):
1. EfficientNet-B0 (1280-D) [Spatial / Texture]
2. Video Swin Transformer (768-D) [Temporal / Motion]
3. Face/Mouth ResNet-18 (512-D) [Facial Articulation / Mouth ROI]

Audio Branch (3 Models):
4. Whisper (512-D) [Semantic Speech]
5. Wav2Vec2 (768-D) [Acoustic Speech]
6. Spectrogram CNN (256-D) [Time-Frequency Acoustic Spectrogram]

Pipeline:
Vision 3V (1280 + 768 + 512) -> Vision Aggregator (2560 -> 1280) -> Vision Projection (256-D)
Audio 3A (512 + 768 + 256)   -> Audio Aggregator (1536 -> 512)  -> Audio Projection (256-D)
                                            │
                             Adaptive Multimodal Fusion (512-D)
                                            │
                                   Deepfake Classifier
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Optional
import torch
import torch.nn as nn

from models.vision_encoder_factory import VisionEncoderFactory
from models.vision_feature_aggregation import VisionFeatureAggregation
from models.face_mouth_encoder import FaceMouthEncoder
from models.whisper_encoder import WhisperEncoder
from models.wav2vec2_encoder import Wav2Vec2Encoder
from models.spectrogram_cnn import SpectrogramCNNEncoder
from models.audio_feature_aggregation import AudioFeatureAggregation
from models.projection_heads import (
    VisionProjectionHead,
    AudioProjectionHead,
)
from models.contrastive_module import ContrastiveModule
from models.adaptive_fusion import AdaptiveFusion
from models.classifier import DeepfakeClassifier


class MultimodalDeepfakeDetector(nn.Module):

    def __init__(
        self,
        enable_efficientnet: bool = True,
        enable_video_swin: bool = True,
        enable_face_mouth: bool = True,
        enable_whisper: bool = True,
        enable_wav2vec2: bool = True,
        enable_spectrogram_cnn: bool = True,
        freeze_backbones: bool = True,
        freeze_strategy: str = "full",
    ):
        super().__init__()

        self.enable_efficientnet = enable_efficientnet
        self.enable_video_swin = enable_video_swin
        self.enable_face_mouth = enable_face_mouth
        self.enable_whisper = enable_whisper
        self.enable_wav2vec2 = enable_wav2vec2
        self.enable_spectrogram_cnn = enable_spectrogram_cnn
        self.freeze_backbones = freeze_backbones
        self.freeze_strategy = freeze_strategy

        # ----------------------------------------
        # Vision Encoders via Factory
        # ----------------------------------------
        if enable_efficientnet:
            self.vision_encoder = VisionEncoderFactory.create(
                "efficientnet",
                freeze_backbone=freeze_backbones,
                freeze_strategy=freeze_strategy,
            )
        else:
            self.vision_encoder = None

        if enable_video_swin:
            self.video_swin_encoder = VisionEncoderFactory.create(
                "video_swin",
                freeze_backbone=freeze_backbones,
                freeze_strategy=freeze_strategy,
            )
        else:
            self.video_swin_encoder = None

        if enable_face_mouth:
            self.face_mouth_encoder = VisionEncoderFactory.create(
                "face_mouth",
                freeze_backbone=freeze_backbones,
                freeze_strategy=freeze_strategy,
            )
        else:
            self.face_mouth_encoder = None

        # Vision Feature Aggregation (Concat 1280 + 768 + 512 = 2560 -> Linear 1280)
        self.vision_aggregator = VisionFeatureAggregation(
            efficientnet_dim=1280,
            video_swin_dim=768,
            face_mouth_dim=512,
            output_dim=1280,
        )

        # ----------------------------------------
        # Audio Encoders
        # ----------------------------------------
        if enable_whisper:
            self.audio_encoder = WhisperEncoder(freeze=freeze_backbones, freeze_strategy=freeze_strategy)
        else:
            self.audio_encoder = None

        if enable_wav2vec2:
            self.wav2vec2_encoder = Wav2Vec2Encoder(freeze=freeze_backbones, freeze_strategy=freeze_strategy)
        else:
            self.wav2vec2_encoder = None

        if enable_spectrogram_cnn:
            # Spectrogram CNN is newly initialized and trainable (freeze=False by default)
            self.spectrogram_cnn_encoder = SpectrogramCNNEncoder(freeze=False)
        else:
            self.spectrogram_cnn_encoder = None

        # Audio Feature Aggregation (Concat 512 + 768 + 256 = 1536 -> Linear 512)
        self.audio_aggregator = AudioFeatureAggregation(
            whisper_dim=512,
            wav2vec2_dim=768,
            spectrogram_dim=256,
            output_dim=512,
        )

        # ----------------------------------------
        # Projection Heads
        # ----------------------------------------
        self.vision_projection = VisionProjectionHead(input_dim=1280)
        self.audio_projection = AudioProjectionHead()

        # ----------------------------------------
        # Contrastive Learning & Adaptive Fusion
        # ----------------------------------------
        self.contrastive = ContrastiveModule()
        self.fusion = AdaptiveFusion()

        # ----------------------------------------
        # Classification Head
        # ----------------------------------------
        self.classifier = DeepfakeClassifier()

    def forward(
        self,
        images: torch.Tensor,
        waveform: Optional[torch.Tensor] = None,
        audio_available: Optional[bool] = None,
        return_loss: bool = True,
        return_intermediate_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images : torch.Tensor
            Shape (B, T, C, H, W)
        waveform : torch.Tensor, optional
            Shape (B, samples) or None
        audio_available : bool, optional
            Explicit boolean flag indicating if audio can be used
        return_loss : bool
            Whether to compute contrastive loss
        return_intermediate_features : bool
            Whether to return intermediate visual/audio feature tensors
        """
        if images.ndim != 5:
            raise ValueError(f"Expected images of shape (B,T,C,H,W), got {tuple(images.shape)}")

        # ----------------------------------------
        # Determine Audio Availability
        # ----------------------------------------
        if audio_available is None:
            if waveform is None or waveform.numel() == 0 or (waveform.ndim == 2 and waveform.shape[-1] == 0):
                audio_available = False
            elif torch.max(torch.abs(waveform)).item() < 1e-7:
                audio_available = False
            else:
                audio_available = True

        if audio_available and waveform is not None and waveform.ndim != 2:
            raise ValueError(f"Expected waveform of shape (B,samples), got {tuple(waveform.shape)}")

        # ----------------------------------------
        # Vision Branch (EfficientNet + Swin + Face/Mouth)
        # ----------------------------------------
        eff_features = None
        swin_features = None
        face_mouth_features = None

        if self.enable_efficientnet and self.vision_encoder is not None:
            eff_features = self.vision_encoder(images)  # (B, 1280)

        if self.enable_video_swin and self.video_swin_encoder is not None:
            swin_features = self.video_swin_encoder(images)  # (B, 768)

        if self.enable_face_mouth and self.face_mouth_encoder is not None:
            face_mouth_features = self.face_mouth_encoder(images)  # (B, 512)

        # Vision Feature Aggregation (Concat 2560 -> Linear 1280)
        fused_vision_features = self.vision_aggregator(
            efficientnet_features=eff_features,
            video_swin_features=swin_features,
            face_mouth_features=face_mouth_features,
        )

        # Vision Projection Head -> (B, 256)
        vision_embedding = self.vision_projection(fused_vision_features)

        # ----------------------------------------
        # Audio Branch (Whisper + Wav2Vec2 + Spectrogram CNN)
        # ----------------------------------------
        whisper_features = None
        wav2vec2_features = None
        spectrogram_features = None
        fused_audio_features = None
        audio_embedding = None
        contrastive_loss = None
        similarity = None

        if audio_available and waveform is not None:
            if self.enable_whisper and self.audio_encoder is not None:
                whisper_features = self.audio_encoder(waveform)  # (B, 512)

            if self.enable_wav2vec2 and self.wav2vec2_encoder is not None:
                wav2vec2_features = self.wav2vec2_encoder(waveform)  # (B, 768)

            if self.enable_spectrogram_cnn and self.spectrogram_cnn_encoder is not None:
                spectrogram_features = self.spectrogram_cnn_encoder(waveform)  # (B, 256)

            # Audio Feature Aggregation (Concat 1536 -> Linear 512)
            fused_audio_features = self.audio_aggregator(
                whisper_features=whisper_features,
                wav2vec2_features=wav2vec2_features,
                spectrogram_features=spectrogram_features,
            )

            if fused_audio_features is not None:
                audio_embedding = self.audio_projection(fused_audio_features)

                if return_loss:
                    contrastive_loss, similarity = self.contrastive(
                        vision_embedding,
                        audio_embedding,
                    )

        # ----------------------------------------
        # Adaptive Fusion
        # ----------------------------------------
        fused_embedding, attention = self.fusion(
            vision_embedding,
            audio_embedding,
            audio_available=audio_available,
        )

        # ----------------------------------------
        # Classification
        # ----------------------------------------
        logits = self.classifier(fused_embedding)

        fusion_mode = "Adaptive Multimodal (3V3)" if audio_available else "Vision Only"

        results = {
            "logits": logits,
            "vision_embedding": vision_embedding,
            "audio_embedding": audio_embedding,
            "fused_embedding": fused_embedding,
            "attention": attention,
            "contrastive_loss": contrastive_loss,
            "similarity": similarity,
            "audio_available": audio_available,
            "vision_weight": attention[:, 0],
            "audio_weight": attention[:, 1],
            "fusion_mode": fusion_mode,
        }

        if return_intermediate_features:
            results["efficientnet_features"] = eff_features
            results["video_swin_features"] = swin_features
            results["face_mouth_features"] = face_mouth_features
            results["fused_vision_features"] = fused_vision_features
            results["whisper_features"] = whisper_features
            results["wav2vec2_features"] = wav2vec2_features
            results["spectrogram_features"] = spectrogram_features
            results["fused_audio_features"] = fused_audio_features

        return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing MultimodalDeepfakeDetector with FULL 3-Vision + 3-Audio (6 Models)...")
    model = MultimodalDeepfakeDetector(
        enable_efficientnet=True,
        enable_video_swin=True,
        enable_face_mouth=True,
        enable_whisper=True,
        enable_wav2vec2=True,
        enable_spectrogram_cnn=True,
        freeze_backbones=True,
    ).to(device)

    images = torch.randn(2, 16, 3, 224, 224, device=device)
    waveform = torch.randn(2, 160000, device=device)

    with torch.no_grad():
        output = model(images, waveform, return_intermediate_features=True)

    print("-" * 60)
    for key, value in output.items():
        if value is None:
            print(f"{key:<30}: None")
        elif isinstance(value, torch.Tensor):
            print(f"{key:<30}: {tuple(value.shape)}")
        else:
            print(f"{key:<30}: {value}")
    print("-" * 60)