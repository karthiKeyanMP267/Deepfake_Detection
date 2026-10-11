"""
models/classifier.py

Deepfake Classification Head

Input:
    (B, 512)

Output:
    Logits:
        (B, 2)

Classes:
    0 -> Real
    1 -> Fake
"""

import torch
import torch.nn as nn


class DeepfakeClassifier(nn.Module):
    """
    Classification head for multimodal deepfake detection.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.BatchNorm1d(hidden_dim),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.BatchNorm1d(hidden_dim // 2),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim // 2,
                num_classes,
            )

        )

    def forward(self, fused_embedding):

        logits = self.classifier(fused_embedding)

        return logits


if __name__ == "__main__":

    batch_size = 8

    fused = torch.randn(
        batch_size,
        512,
    )

    model = DeepfakeClassifier()

    logits = model(fused)

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    print("-" * 60)
    print("Input Shape :", fused.shape)
    print("Logits Shape:", logits.shape)
    print("Prediction  :", prediction)
    print("-" * 60)