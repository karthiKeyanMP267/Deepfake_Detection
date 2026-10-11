"""
models/projection_heads.py

Projection heads for cross-modal representation learning.

Vision:
    1280 -> 512 -> 256

Audio:
    512 -> 512 -> 256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Generic projection head.

    Linear
        ↓
    BatchNorm
        ↓
    ReLU
        ↓
    Dropout
        ↓
    Linear
        ↓
    L2 Normalize
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.network(x)

        # Normalize embeddings for contrastive learning
        x = F.normalize(x, dim=1)

        return x


class VisionProjectionHead(nn.Module):
    """
    Projection head for unified vision features.
    """

    def __init__(self, input_dim: int = 1280):

        super().__init__()

        self.projector = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=512,
            output_dim=256,
        )

    def forward(self, features):

        return self.projector(features)


class AudioProjectionHead(nn.Module):
    """
    Projection head for Whisper features.
    """

    def __init__(self):

        super().__init__()

        self.projector = ProjectionHead(
            input_dim=512,
            hidden_dim=512,
            output_dim=256,
        )

    def forward(self, features):

        return self.projector(features)


if __name__ == "__main__":

    vision = VisionProjectionHead()
    audio = AudioProjectionHead()

    vision_input = torch.randn(4, 1280)
    audio_input = torch.randn(4, 512)

    vision_embedding = vision(vision_input)
    audio_embedding = audio(audio_input)

    print("-" * 60)
    print("Vision Input :", vision_input.shape)
    print("Vision Output:", vision_embedding.shape)
    print()

    print("Audio Input  :", audio_input.shape)
    print("Audio Output :", audio_embedding.shape)
    print("-" * 60)