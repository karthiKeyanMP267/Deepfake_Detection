"""
models/vision_feature_aggregation.py

Vision Feature Aggregation Module for 3-Vision Encoders:
- EfficientNet-B0 (1280-D) [Texture / Spatial]
- Video Swin Transformer (768-D) [Temporal / Motion]
- Face/Mouth ResNet-18 (512-D) [Facial Articulation / Mouth ROI]

Concatenates available visual features and projects through a single linear layer:
(1280 + 768 + 512 = 2560-D) -> Single Linear Layer -> Unified Vision Feature (1280-D)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional
import torch
import torch.nn as nn


class VisionFeatureAggregation(nn.Module):
    """
    Combines texture features (EfficientNet-B0), temporal features (Video Swin),
    and facial ROI features (Face/Mouth ResNet-18) via concatenation and linear projection.
    """

    def __init__(
        self,
        efficientnet_dim: int = 1280,
        video_swin_dim: int = 768,
        face_mouth_dim: int = 512,
        output_dim: int = 1280,
    ):
        super().__init__()

        self.efficientnet_dim = efficientnet_dim
        self.video_swin_dim = video_swin_dim
        self.face_mouth_dim = face_mouth_dim
        self.output_dim = output_dim

        # 3-Vision Concatenation Dim (1280 + 768 + 512 = 2560)
        concatenated_dim_3v = efficientnet_dim + video_swin_dim + face_mouth_dim
        self.fc_3v = nn.Linear(concatenated_dim_3v, output_dim)

        # 2-Vision Concatenation Dim (1280 + 768 = 2048) for backward compatibility
        concatenated_dim_2v = efficientnet_dim + video_swin_dim
        self.fc_2v = nn.Linear(concatenated_dim_2v, output_dim)
        # Keep self.fc reference for backward compatibility with 2v checkpoint
        self.fc = self.fc_2v

    def forward(
        self,
        efficientnet_features: Optional[torch.Tensor] = None,
        video_swin_features: Optional[torch.Tensor] = None,
        face_mouth_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        efficientnet_features : torch.Tensor, optional
            Shape: (B, 1280)
        video_swin_features : torch.Tensor, optional
            Shape: (B, 768)
        face_mouth_features : torch.Tensor, optional
            Shape: (B, 512)

        Returns
        -------
        torch.Tensor
            Shape: (B, 1280)
        """
        active_features = []
        if efficientnet_features is not None:
            active_features.append(efficientnet_features)
        if video_swin_features is not None:
            active_features.append(video_swin_features)
        if face_mouth_features is not None:
            active_features.append(face_mouth_features)

        if not active_features:
            raise ValueError("At least one vision feature tensor must be provided.")

        # 3-Vision Model Mode
        if efficientnet_features is not None and video_swin_features is not None and face_mouth_features is not None:
            concatenated = torch.cat([efficientnet_features, video_swin_features, face_mouth_features], dim=1)
            return self.fc_3v(concatenated)

        # 2-Vision Model Mode (EfficientNet + Swin)
        if efficientnet_features is not None and video_swin_features is not None:
            concatenated = torch.cat([efficientnet_features, video_swin_features], dim=1)
            return self.fc_2v(concatenated)

        # Single feature fallback
        if len(active_features) == 1:
            feat = active_features[0]
            if feat.shape[1] == self.output_dim:
                return feat

        # Fallback padding to match output_dim
        concatenated = torch.cat(active_features, dim=1)
        if hasattr(self, 'fc_3v') and concatenated.shape[1] == self.fc_3v.in_features:
            return self.fc_3v(concatenated)
        elif hasattr(self, 'fc_2v') and concatenated.shape[1] == self.fc_2v.in_features:
            return self.fc_2v(concatenated)
        else:
            # Dynamically create fallback projection if unexpected shape
            fallback_proj = nn.Linear(concatenated.shape[1], self.output_dim).to(concatenated.device)
            return fallback_proj(concatenated)


if __name__ == "__main__":
    eff_feat = torch.randn(4, 1280)
    swin_feat = torch.randn(4, 768)
    fm_feat = torch.randn(4, 512)

    aggregator = VisionFeatureAggregation()
    output_3v = aggregator(eff_feat, swin_feat, fm_feat)
    output_2v = aggregator(eff_feat, swin_feat)

    print("-" * 60)
    print("3-Vision Input Shapes :", eff_feat.shape, swin_feat.shape, fm_feat.shape)
    print("3-Vision Aggregated   :", output_3v.shape)
    print("2-Vision Aggregated   :", output_2v.shape)
    print("-" * 60)
