"""
models/audio_feature_aggregation.py

Audio Feature Aggregation Module for 3-Audio Encoders:
- Whisper (512-D) [Semantic Speech]
- Wav2Vec2 (768-D) [Acoustic Speech]
- Spectrogram CNN (256-D) [Time-Frequency Acoustic Spectrogram]

Concatenates available audio features and projects through a single linear layer:
(512 + 768 + 256 = 1536-D) -> Single Linear Layer -> Unified Audio Feature (512-D)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional
import torch
import torch.nn as nn


class AudioFeatureAggregation(nn.Module):
    """
    Combines Whisper semantic features, Wav2Vec2 acoustic features,
    and Spectrogram CNN time-frequency features via concatenation and linear projection.
    """

    def __init__(
        self,
        whisper_dim: int = 512,
        wav2vec2_dim: int = 768,
        spectrogram_dim: int = 256,
        output_dim: int = 512,
    ):
        super().__init__()

        self.whisper_dim = whisper_dim
        self.wav2vec2_dim = wav2vec2_dim
        self.spectrogram_dim = spectrogram_dim
        self.output_dim = output_dim

        # 3-Audio Concatenation Dim (512 + 768 + 256 = 1536)
        concatenated_dim_3a = whisper_dim + wav2vec2_dim + spectrogram_dim
        self.fc_3a = nn.Linear(concatenated_dim_3a, output_dim)

        # 2-Audio Concatenation Dim (512 + 768 = 1280) for backward compatibility
        concatenated_dim_2a = whisper_dim + wav2vec2_dim
        self.fc_2a = nn.Linear(concatenated_dim_2a, output_dim)
        # Keep self.fc reference for backward compatibility
        self.fc = self.fc_2a

    def forward(
        self,
        whisper_features: Optional[torch.Tensor] = None,
        wav2vec2_features: Optional[torch.Tensor] = None,
        spectrogram_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        whisper_features : torch.Tensor, optional
            Shape: (B, 512)
        wav2vec2_features : torch.Tensor, optional
            Shape: (B, 768)
        spectrogram_features : torch.Tensor, optional
            Shape: (B, 256)

        Returns
        -------
        torch.Tensor
            Shape: (B, 512)
        """
        active_features = []
        if whisper_features is not None:
            active_features.append(whisper_features)
        if wav2vec2_features is not None:
            active_features.append(wav2vec2_features)
        if spectrogram_features is not None:
            active_features.append(spectrogram_features)

        if not active_features:
            raise ValueError("At least one audio feature tensor must be provided.")

        # 3-Audio Model Mode
        if whisper_features is not None and wav2vec2_features is not None and spectrogram_features is not None:
            concatenated = torch.cat([whisper_features, wav2vec2_features, spectrogram_features], dim=1)
            return self.fc_3a(concatenated)

        # 2-Audio Model Mode (Whisper + Wav2Vec2)
        if whisper_features is not None and wav2vec2_features is not None:
            concatenated = torch.cat([whisper_features, wav2vec2_features], dim=1)
            return self.fc_2a(concatenated)

        # Single feature fallback
        if len(active_features) == 1:
            feat = active_features[0]
            if feat.shape[1] == self.output_dim:
                return feat

        # Fallback padding to match output_dim
        concatenated = torch.cat(active_features, dim=1)
        if hasattr(self, 'fc_3a') and concatenated.shape[1] == self.fc_3a.in_features:
            return self.fc_3a(concatenated)
        elif hasattr(self, 'fc_2a') and concatenated.shape[1] == self.fc_2a.in_features:
            return self.fc_2a(concatenated)
        else:
            fallback_proj = nn.Linear(concatenated.shape[1], self.output_dim).to(concatenated.device)
            return fallback_proj(concatenated)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    w_feat = torch.randn(4, 512, device=device)
    w2v_feat = torch.randn(4, 768, device=device)
    spec_feat = torch.randn(4, 256, device=device)

    aggregator = AudioFeatureAggregation().to(device)

    fused_3a = aggregator(w_feat, w2v_feat, spec_feat)
    fused_2a = aggregator(w_feat, w2v_feat)
    fallback_1a = aggregator(w_feat)

    print("=" * 60)
    print("Audio Feature Aggregation 3A Test")
    print("3-Audio Input Shapes :", w_feat.shape, w2v_feat.shape, spec_feat.shape)
    print("3-Audio Aggregated   :", fused_3a.shape)
    print("2-Audio Aggregated   :", fused_2a.shape)
    print("1-Audio Fallback     :", fallback_1a.shape)
    print("=" * 60)
