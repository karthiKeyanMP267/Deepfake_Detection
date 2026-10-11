"""
models/efficientnet.py

Visual encoder for multimodal deepfake detection.

Input:
    (B, T, 3, 224, 224)

Output:
    (B, 1280)
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from models.base_vision_encoder import BaseVisionEncoder


class EfficientNetEncoder(BaseVisionEncoder):
    """
    EfficientNet-B0 backbone for extracting video-level visual embeddings.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_strategy: str = "full",
        pooling: str = "mean",
    ):
        super().__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        backbone = efficientnet_b0(weights=weights)

        # Feature extractor
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        self.feature_dim = 1280
        self.pooling = pooling.lower()

        # Handle freezing strategies
        if freeze_strategy == "partial":
            for i, child in enumerate(self.features.children()):
                if i < 7:  # Freeze features[0..6]
                    for param in child.parameters():
                        param.requires_grad = False
                else:      # Train features[7..8]
                    for param in child.parameters():
                        param.requires_grad = True
        elif freeze_strategy == "full" or freeze_backbone:
            for parameter in self.features.parameters():
                parameter.requires_grad = False
        elif freeze_strategy == "none":
            for parameter in self.features.parameters():
                parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:
                (B, T, 3, 224, 224)

        Returns
        -------
        torch.Tensor
            Shape:
                (B, 1280)
        """

        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape (B,T,C,H,W), got {tuple(x.shape)}"
            )

        batch_size, num_frames, channels, height, width = x.shape

        # Merge batch and time
        x = x.view(batch_size * num_frames, channels, height, width)

        # EfficientNet feature extraction
        x = self.features(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # Restore time dimension
        x = x.view(batch_size, num_frames, self.feature_dim)

        # Temporal pooling
        if self.pooling == "mean":
            x = x.mean(dim=1)

        elif self.pooling == "max":
            x, _ = x.max(dim=1)

        else:
            raise ValueError(
                "pooling must be either 'mean' or 'max'"
            )

        return x

    @property
    def output_dim(self) -> int:
        """
        Returns feature dimension.
        """
        return self.feature_dim


if __name__ == "__main__":

    model = EfficientNetEncoder(
        pretrained=True,
        freeze_backbone=False,
        pooling="mean",
    )

    dummy = torch.randn(2, 16, 3, 224, 224)

    output = model(dummy)

    print("-" * 50)
    print("Input Shape :", dummy.shape)
    print("Output Shape:", output.shape)
    print("Feature Dim :", model.output_dim)
    print("-" * 50)