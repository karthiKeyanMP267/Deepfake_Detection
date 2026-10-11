"""
models/video_swin.py

Video Swin Transformer Encoder for temporal visual feature extraction.

Input:
    (B, T, C, H, W)

Output:
    (B, 768)
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models
from torchvision.models.video import Swin3D_T_Weights

from models.base_vision_encoder import BaseVisionEncoder


class VideoSwinEncoder(BaseVisionEncoder):
    """
    Video Swin Transformer (Swin3D-Tiny) backbone for spatio-temporal feature extraction.
    Captures motion consistency, frame-to-frame coherence, flickering, and temporal artifacts.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_strategy: str = "full",
    ):
        super().__init__()

        self._output_dim = 768

        try:
            if pretrained:
                weights = Swin3D_T_Weights.DEFAULT
                self.backbone = video_models.swin3d_t(weights=weights)
            else:
                self.backbone = video_models.swin3d_t(weights=None)
        except Exception as e:
            # Fallback if download/network fails or offline
            print(f"[VideoSwinEncoder] Warning loading weights ({e}). Initializing without pretrained weights.")
            self.backbone = video_models.swin3d_t(weights=None)

        # Replace classification head with Identity to extract 768-dim features
        self.backbone.head = nn.Identity()

        if freeze_strategy == "partial":
            if hasattr(self.backbone, "features"):
                for i, child in enumerate(self.backbone.features.children()):
                    if i < 6:  # Freeze features[0..5]
                        for param in child.parameters():
                            param.requires_grad = False
                    else:      # Train features[6]
                        for param in child.parameters():
                            param.requires_grad = True
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            else:
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = ("6" in name or "norm" in name)
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
            Shape: (B, T, C, H, W)

        Returns
        -------
        torch.Tensor
            Shape: (B, 768)
        """
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {tuple(x.shape)}")

        # Convert (B, T, C, H, W) -> (B, C, T, H, W) for 3D Video Swin
        x_3d = x.permute(0, 2, 1, 3, 4)

        # Extract spatio-temporal features
        features = self.backbone(x_3d)

        return features


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoSwinEncoder(pretrained=False).to(device)
    dummy_input = torch.randn(2, 16, 3, 224, 224, device=device)

    with torch.no_grad():
        output = model(dummy_input)

    print("-" * 50)
    print("Input Shape :", dummy_input.shape)
    print("Output Shape:", output.shape)
    print("Feature Dim :", model.output_dim)
    print("-" * 50)
