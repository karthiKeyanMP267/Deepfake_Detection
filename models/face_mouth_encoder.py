"""
models/face_mouth_encoder.py

Dedicated Face/Mouth-Specific Visual Encoder for Multimodal Deepfake Detection.

This module focuses specifically on the lower facial / mouth ROI (lips, jaw, teeth,
mouth movement, and lower-facial articulation artifacts). It extracts ROI features
from 224x224 input video frames using a lightweight pretrained ResNet-18 backbone.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from models.base_vision_encoder import BaseVisionEncoder


class FaceMouthEncoder(BaseVisionEncoder):
    """
    Lightweight ResNet-18 based Face/Mouth visual branch.
    Crops lower-facial / mouth ROI from input frame sequences and extracts
    512-dimensional visual embeddings.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        freeze_strategy: str = "full",
    ):
        super().__init__()

        self._output_dim = 512

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)

        # Replace FC classification head with Identity to output 512-dim features
        resnet.fc = nn.Identity()
        self.backbone = resnet

        # Freezing logic
        if freeze_strategy == "partial":
            # Freeze layer1 and layer2, unfreeze layer3 and layer4
            for name, param in self.backbone.named_parameters():
                if "layer3" in name or "layer4" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif freeze_strategy == "full" or freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_strategy == "none":
            for param in self.backbone.parameters():
                param.requires_grad = True

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input video frame sequence of shape (B, T, 3, 224, 224)

        Returns
        -------
        torch.Tensor
            Mouth/Lower-face embedding of shape (B, 512)
        """
        if x.ndim != 5:
            raise ValueError(f"Expected input tensor of shape (B, T, C, H, W), got {tuple(x.shape)}")

        B, T, C, H, W = x.shape

        # Extract lower-facial / mouth ROI (lower half of 224x224 face crop: rows 112..224, cols 56..168)
        mouth_roi = x[:, :, :, H // 2: H, W // 4: (3 * W) // 4]  # Shape: (B, T, C, 112, 112)

        # Resize mouth ROI to (224, 224) for standard ResNet receptive field
        mouth_roi_reshaped = mouth_roi.reshape(B * T, C, H // 2, (3 * W) // 4 - W // 4)
        mouth_roi_resized = F.interpolate(
            mouth_roi_reshaped, size=(224, 224), mode="bilinear", align_corners=False
        )  # Shape: (B * T, 3, 224, 224)

        # Extract features through ResNet-18 backbone
        features = self.backbone(mouth_roi_resized)  # Shape: (B * T, 512)

        # Reshape to (B, T, 512) and perform temporal mean pooling across frames
        features_seq = features.view(B, T, 512)
        face_mouth_features = features_seq.mean(dim=1)  # Shape: (B, 512)

        return face_mouth_features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing FaceMouthEncoder...")
    encoder = FaceMouthEncoder(pretrained=False, freeze_backbone=True).to(device)

    dummy_video = torch.randn(2, 16, 3, 224, 224, device=device)
    out = encoder(dummy_video)

    print("-" * 60)
    print("Input Video Shape :", dummy_video.shape)
    print("Output Feature    :", out.shape)
    print("Output Dimension  :", encoder.output_dim)
    print("Frozen Parameters :", all(not p.requires_grad for p in encoder.parameters()))
    print("-" * 60)
