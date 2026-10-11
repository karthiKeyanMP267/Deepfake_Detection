"""
models/adaptive_fusion.py

Adaptive Multimodal Fusion Module

Vision Embedding (256)
            │
            │
            ▼
      Learn Vision Weight
            │
            │
Audio Embedding (256)
            │
            ▼
      Learn Audio Weight
            │
            ▼
Weighted Fusion
            │
            ▼
Output (512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion using learned attention weights.

    Input
    -----
    vision_embedding : (B,256)

    audio_embedding : (B,256)

    Output
    ------
    fused_embedding : (B,512)

    attention_weights : (B,2)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.attention_network = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                2,
            )
        )

    def forward(
        self,
        vision_embedding: torch.Tensor,
        audio_embedding: torch.Tensor = None,
        audio_available: bool = True,
    ):
        if not audio_available or audio_embedding is None:
            B = vision_embedding.shape[0]
            device = vision_embedding.device

            # Mode 2: Vision-Only Mode
            weighted_vision = vision_embedding
            weighted_audio = torch.zeros_like(vision_embedding)
            fused_embedding = torch.cat([weighted_vision, weighted_audio], dim=1)

            # Vision Weight = 1.0, Audio Weight = 0.0
            weights = torch.tensor([[1.0, 0.0]], dtype=vision_embedding.dtype, device=device).repeat(B, 1)
            return fused_embedding, weights

        if vision_embedding.shape != audio_embedding.shape:

            raise ValueError(
                "Vision and Audio embeddings must have identical shapes."
            )

        combined = torch.cat(
            [
                vision_embedding,
                audio_embedding,
            ],
            dim=1,
        )

        weights = self.attention_network(combined)

        weights = F.softmax(
            weights,
            dim=1,
        )

        vision_weight = weights[:, 0].unsqueeze(1)

        audio_weight = weights[:, 1].unsqueeze(1)

        weighted_vision = (
            vision_weight *
            vision_embedding
        )

        weighted_audio = (
            audio_weight *
            audio_embedding
        )

        fused_embedding = torch.cat(
            [
                weighted_vision,
                weighted_audio,
            ],
            dim=1,
        )

        return fused_embedding, weights


if __name__ == "__main__":

    batch_size = 4

    vision = torch.randn(
        batch_size,
        256,
    )

    audio = torch.randn(
        batch_size,
        256,
    )

    model = AdaptiveFusion()

    fused, weights = model(
        vision,
        audio,
    )

    print("-" * 60)
    print("Vision Shape :", vision.shape)
    print("Audio Shape  :", audio.shape)
    print("Fused Shape  :", fused.shape)
    print("Weights Shape:", weights.shape)
    print()
    print("Sample Attention Weights")
    print(weights)
    print("-" * 60)