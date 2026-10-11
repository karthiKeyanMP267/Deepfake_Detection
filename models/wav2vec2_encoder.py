"""
models/wav2vec2_encoder.py

Wav2Vec2 Audio Encoder for Multimodal Deepfake Detection.

Input:
    Raw audio waveform
    Shape: (B, samples) @ 16,000 Hz sample rate

Output:
    Audio embedding
    Shape: (B, 768)
"""

import torch
import torch.nn as nn
import torchaudio


class Wav2Vec2Encoder(nn.Module):
    """
    Wav2Vec2 Audio Feature Extractor using torchaudio.pipelines.WAV2VEC2_BASE
    """

    def __init__(self, freeze: bool = True, freeze_strategy: str = "full"):
        super().__init__()

        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.sample_rate = self.bundle.sample_rate  # 16000 Hz

        # Load pre-trained Wav2Vec2 base model
        self.model = self.bundle.get_model()

        # Output feature dimension of the last transformer layer (768 for Wav2Vec2 Base)
        self.feature_dim = 768

        if freeze_strategy == "partial":
            # Freeze feature_extractor, feature_projection, pos_conv_embed, layer_norm
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.model.encoder.feature_projection.parameters():
                param.requires_grad = False
            for param in self.model.encoder.transformer.pos_conv_embed.parameters():
                param.requires_grad = False
            for param in self.model.encoder.transformer.layer_norm.parameters():
                param.requires_grad = False

            # Freeze Transformer layers 0..7, train 8..11
            for i, layer in enumerate(self.model.encoder.transformer.layers):
                if i < 8:  # Freeze layers 0..7
                    for param in layer.parameters():
                        param.requires_grad = False
                else:      # Train layers 8..11
                    for param in layer.parameters():
                        param.requires_grad = True
        elif freeze_strategy == "full" or freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze_strategy == "none":
            for param in self.model.parameters():
                param.requires_grad = True

    @property
    def output_dim(self) -> int:
        return self.feature_dim

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : torch.Tensor
            Raw audio waveform of shape (B, samples)

        Returns
        -------
        torch.Tensor
            Wav2Vec2 global pooled embedding of shape (B, 768)
        """
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Input waveform must be a torch.Tensor.")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape (B, samples), got {tuple(waveform.shape)}")

        # Extract features from pre-trained Wav2Vec2 model
        # model.extract_features returns (list_of_layer_outputs, lengths)
        features, _ = self.model.extract_features(waveform)

        # Use the output from the last transformer layer (layer 12)
        last_layer_features = features[-1]  # Shape: (B, T, 768)

        # Global Average Pooling across the temporal dimension T
        embedding = last_layer_features.mean(dim=1)  # Shape: (B, 768)

        return embedding


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav2vec2 = Wav2Vec2Encoder(freeze=True).to(device)

    # 10 seconds of audio @ 16kHz
    dummy_waveform = torch.randn(2, 160000, device=device)

    output = wav2vec2(dummy_waveform)

    print("=" * 60)
    print("Wav2Vec2 Encoder Smoke Test")
    print("Sample Rate :", wav2vec2.sample_rate)
    print("Input Shape :", dummy_waveform.shape)
    print("Output Shape:", output.shape)
    print("Output Dim  :", wav2vec2.output_dim)
    print("=" * 60)
