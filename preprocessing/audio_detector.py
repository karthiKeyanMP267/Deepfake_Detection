"""
preprocessing/audio_detector.py

Robust Audio Availability Detector for Multimodal Deepfake Detection.

Checks audio files and waveforms for:
- Missing files / paths
- File corruption / unreadable formats
- Zero duration / empty streams
- All-zero silence / NaN / Inf values
"""

import os
from pathlib import Path
from typing import Tuple, Union
import torch
import torchaudio


class AudioDetector:
    """
    Audio Availability Checker.
    Inspects audio files and loaded waveforms to determine if audio modality can be processed.
    """

    def __init__(self, sample_rate: int = 16000, silence_threshold: float = 1e-7):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold

    def check_audio_file(self, audio_path: Union[str, Path, None]) -> Tuple[bool, str]:
        """
        Check if an audio file exists, is non-empty, and can be read by torchaudio.

        Returns
        -------
        (audio_available: bool, reason: str)
        """
        if audio_path is None or str(audio_path).strip() == "":
            return False, "Audio path is None or empty string"

        path = Path(audio_path)
        if not path.exists():
            return False, f"Audio file does not exist: {path}"

        if not path.is_file():
            return False, f"Audio path is not a valid file: {path}"

        try:
            if os.path.getsize(path) == 0:
                return False, f"Audio file is 0 bytes: {path}"
        except OSError as e:
            return False, f"Audio file stat failure: {e}"

        # Attempt reading audio file
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            return False, f"Audio loading failure (corrupted/unreadable): {str(e)}"

        return self.check_waveform(waveform, sr)

    def check_waveform(self, waveform: torch.Tensor, sample_rate: int = 16000) -> Tuple[bool, str]:
        """
        Inspect a PyTorch waveform tensor for validity.

        Returns
        -------
        (audio_available: bool, reason: str)
        """
        if waveform is None:
            return False, "Waveform is None"

        if not isinstance(waveform, torch.Tensor):
            return False, f"Expected torch.Tensor, got {type(waveform)}"

        if waveform.numel() == 0 or waveform.shape[-1] == 0:
            return False, "Empty audio waveform (duration == 0)"

        if torch.isnan(waveform).any():
            return False, "Audio waveform contains NaN values"

        if torch.isinf(waveform).any():
            return False, "Audio waveform contains Inf values"

        max_amplitude = torch.max(torch.abs(waveform)).item()
        if max_amplitude < self.silence_threshold:
            return False, f"Empty audio stream (silent/zero amplitude, max_amp={max_amplitude:.2e})"

        return True, f"Usable audio stream detected (samples={waveform.shape[-1]}, sr={sample_rate})"


# Convenience standalone function
def check_audio_availability(audio_path: Union[str, Path, None], sample_rate: int = 16000) -> Tuple[bool, str]:
    detector = AudioDetector(sample_rate=sample_rate)
    return detector.check_audio_file(audio_path)


if __name__ == "__main__":
    detector = AudioDetector()
    print("Testing AudioDetector with None:", detector.check_audio_file(None))
    print("Testing AudioDetector with missing file:", detector.check_audio_file("non_existent_audio.wav"))
    print("Testing AudioDetector with empty tensor:", detector.check_waveform(torch.zeros(0)))
    print("Testing AudioDetector with silent waveform:", detector.check_waveform(torch.zeros(16000)))
    print("Testing AudioDetector with valid waveform:", detector.check_waveform(torch.randn(16000)))
