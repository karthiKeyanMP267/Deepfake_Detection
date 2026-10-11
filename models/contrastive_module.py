"""
models/contrastive_module.py

Cross-modal contrastive learning using InfoNCE loss.

Image Embeddings
        │
        ├─────────────┐
        │             │
        ▼             ▼
 Normalize       Normalize
        │             │
        └─────Similarity─────┐
                             ▼
                        InfoNCE Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModule(nn.Module):
    """
    Cross-modal InfoNCE loss.

    Input:
        image_embeddings : (B, 256)
        audio_embeddings : (B, 256)

    Output:
        loss : scalar
        similarity_matrix : (B, B)
    """

    def __init__(
        self,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        image_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
    ):
        """
        Parameters
        ----------
        image_embeddings : (B, 256)

        audio_embeddings : (B, 256)

        Returns
        -------
        loss
        similarity_matrix
        """

        if image_embeddings.shape != audio_embeddings.shape:
            raise ValueError(
                "Image and audio embeddings must have identical shapes."
            )

        # L2 Normalize
        image_embeddings = F.normalize(image_embeddings, dim=1)
        audio_embeddings = F.normalize(audio_embeddings, dim=1)

        # Similarity matrix
        similarity = (
            image_embeddings @ audio_embeddings.T
        ) / self.temperature

        batch_size = similarity.size(0)

        labels = torch.arange(
            batch_size,
            device=similarity.device,
        )

        # Image → Audio
        loss_i2a = F.cross_entropy(
            similarity,
            labels,
        )

        # Audio → Image
        loss_a2i = F.cross_entropy(
            similarity.T,
            labels,
        )

        loss = (loss_i2a + loss_a2i) / 2

        return loss, similarity


if __name__ == "__main__":

    batch_size = 8

    image = torch.randn(batch_size, 256)
    audio = torch.randn(batch_size, 256)

    model = ContrastiveModule()

    loss, similarity = model(image, audio)

    print("-" * 60)
    print("Image Embeddings :", image.shape)
    print("Audio Embeddings :", audio.shape)
    print("Similarity Matrix:", similarity.shape)
    print("Contrastive Loss :", loss.item())
    print("-" * 60)