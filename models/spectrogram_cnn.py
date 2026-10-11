"""
models/spectrogram_cnn.py

Spectrogram-Based Audio CNN Branch for Multimodal Deepfake Detection.

This module converts raw 16kHz audio waveforms into log-Mel Spectrograms
(time-frequency acoustic representations) and extracts 256-dimensional embeddings
using a lightweight 2D Convolutional Neural Network (CNN).

Spectrogram Parameters:
- Sample Rate : 16,000 Hz
- FFT Size    : 1,024
- Hop Length  : 512
- Mel Bins    : 80
- Output Dim  : 256
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpectrogramCNNEncoder(nn.Module):
    """
    Lightweight 2D CNN operating on log-Mel Spectrograms to extract
    256-dimensional time-frequency acoustic embeddings.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 80,
        output_dim: int = 256,
        freeze: bool = False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self._output_dim = output_dim

        # Mel Spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Lightweight 2D CNN feature extractor
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Final FC projection layer (256 -> 256)
        self.fc = nn.Linear(256, output_dim)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : torch.Tensor
            Raw audio waveform of shape (B, samples)

        Returns
        -------
        torch.Tensor
            Spectrogram CNN embedding of shape (B, 256)
        """
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform of shape (B, samples), got {tuple(waveform.shape)}")

        # Ensure mel_transform is on correct device
        if next(self.parameters()).device != waveform.device:
            self.to(waveform.device)

        # 1. Compute Mel Spectrogram: (B, n_mels, time_steps)
        mel_spec = self.mel_transform(waveform)

        # 2. Convert to log scale: log(mel + 1e-6)
        log_mel = torch.log(mel_spec + 1e-6)

        # 3. Add channel dimension: (B, 1, n_mels, time_steps)
        x = log_mel.unsqueeze(1)

        # 4. Pass through 2D CNN blocks -> (B, 256, 1, 1)
        feat_map = self.conv_blocks(x)

        # 5. Flatten & Linear projection -> (B, 256)
        flattened = feat_map.view(feat_map.size(0), -1)
        features = self.fc(flattened)

        return features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing SpectrogramCNNEncoder...")
    encoder = SpectrogramCNNEncoder().to(device)

    dummy_audio = torch.randn(2, 160000, device=device)
    out = encoder(dummy_audio)

    print("-" * 60)
    print("Input Waveform Shape :", dummy_audio.shape)
    print("Output Feature Shape :", out.shape)
    print("Output Dimension     :", encoder.output_dim)
    print("Trainable Parameters :", any(p.requires_grad for p in encoder.parameters()))
    print("-" * 60)
