"""
models/base_vision_encoder.py

Abstract base class for all vision encoders in the modular vision framework.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseVisionEncoder(nn.Module, ABC):
    """
    Abstract interface for visual feature extractors.
    All vision encoders (EfficientNet, Video Swin, MediaPipe Face Mesh, etc.)
    must inherit from this base class.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Returns the output feature dimension of the encoder.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from input video frames.

        Parameters
        ----------
        x : torch.Tensor
            Input video tensor of shape (B, T, C, H, W).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, output_dim).
        """
        pass
