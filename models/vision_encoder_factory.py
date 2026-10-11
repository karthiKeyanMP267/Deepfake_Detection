"""
models/vision_encoder_factory.py

Registry-based Vision Encoder Factory.
Allows modular registration and dynamic creation of vision encoders (EfficientNet, Video Swin,
MediaPipe Face Mesh, ViT, ConvNeXt, etc.) without altering the main multimodal pipeline.
"""

from typing import Dict, Type, Any
import torch.nn as nn

from models.base_vision_encoder import BaseVisionEncoder
from models.efficientnet import EfficientNetEncoder
from models.video_swin import VideoSwinEncoder
from models.face_mouth_encoder import FaceMouthEncoder


class VisionEncoderFactory:
    """
    Registry factory for vision encoders.
    """

    _registry: Dict[str, Type[BaseVisionEncoder]] = {}

    @classmethod
    def register(cls, name: str, encoder_cls: Type[BaseVisionEncoder]):
        """
        Register a vision encoder class under a unique name.
        """
        cls._registry[name.lower()] = encoder_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseVisionEncoder:
        """
        Instantiate a vision encoder by name.
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise KeyError(
                f"Encoder '{name}' is not registered. Registered encoders: {list(cls._registry.keys())}"
            )
        return cls._registry[name_lower](**kwargs)

    @classmethod
    def list_registered(cls) -> list:
        return list(cls._registry.keys())


# Register core encoders
VisionEncoderFactory.register("efficientnet", EfficientNetEncoder)
VisionEncoderFactory.register("video_swin", VideoSwinEncoder)
VisionEncoderFactory.register("face_mouth", FaceMouthEncoder)


if __name__ == "__main__":
    print("Registered encoders:", VisionEncoderFactory.list_registered())
    eff_enc = VisionEncoderFactory.create("efficientnet", pretrained=False)
    swin_enc = VisionEncoderFactory.create("video_swin", pretrained=False)
    print("EfficientNet dim:", eff_enc.output_dim)
    print("Video Swin dim   :", swin_enc.output_dim)
