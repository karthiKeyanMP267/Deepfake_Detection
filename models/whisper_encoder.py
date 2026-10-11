"""
models/whisper_encoder.py

Whisper Audio Encoder

Input:
    Raw waveform
    Shape: (B, samples)

Output:
    Audio embedding
    Shape: (B, 512)
"""

import torch
import torch.nn as nn
import whisper


class WhisperEncoder(nn.Module):
    """
    Whisper Audio Feature Extractor
    """

    def __init__(self, model_name: str = "base", freeze: bool = True, freeze_strategy: str = "full"):
        super().__init__()

        whisper_model = whisper.load_model(model_name, device="cpu")

        # Keep only the encoder
        self.encoder = whisper_model.encoder

        # Feature dimension (512 for base model)
        self.feature_dim = whisper_model.dims.n_audio_state

        if freeze_strategy == "partial":
            # Freeze conv layers and blocks 0..3
            for param in self.encoder.conv1.parameters():
                param.requires_grad = False
            for param in self.encoder.conv2.parameters():
                param.requires_grad = False
            for i, block in enumerate(self.encoder.blocks):
                if i < 4:  # Freeze blocks 0..3
                    for param in block.parameters():
                        param.requires_grad = False
                else:      # Train blocks 4..5
                    for param in block.parameters():
                        param.requires_grad = True
            for param in self.encoder.ln_post.parameters():
                param.requires_grad = True
        elif freeze_strategy == "full" or freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif freeze_strategy == "none":
            for param in self.encoder.parameters():
                param.requires_grad = True

    @property
    def output_dim(self):
        return self.feature_dim

    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert batch of waveforms into Whisper log-Mel spectrograms.

        Parameters
        ----------
        waveform : Tensor
            Shape: (B, samples)

        Returns
        -------
        Tensor
            Shape: (B, 80, 3000)
        """
        mel_list = []

        with torch.no_grad():
            for audio in waveform:
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio)
                mel_list.append(mel)

        return torch.stack(mel_list)

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : Tensor
            Shape: (B, samples)

        Returns
        -------
        Tensor
            Shape: (B, 512)
        """

        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        if waveform.ndim != 2:
            raise ValueError(
                f"Expected input shape (B, samples), got {tuple(waveform.shape)}"
            )

        # Convert waveform -> Whisper log-Mel
        mel = self.waveform_to_mel(waveform)

        # Match model dtype and device
        mel = mel.to(
            device=waveform.device,
            dtype=next(self.parameters()).dtype,
        )

        # Whisper encoder (frozen feature extractor)
        with torch.no_grad():
            features = self.encoder(mel)

        # Global Average Pooling
        embedding = features.mean(dim=1)

        return embedding


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WhisperEncoder("base").to(device)

    # 10 seconds @16kHz
    dummy_waveform = torch.randn(
        2,
        160000,
        device=device,
    )

    output = model(dummy_waveform)

    print("Input Shape :", dummy_waveform.shape)
    print("Output Shape:", output.shape)