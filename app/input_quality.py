"""
app/input_quality.py & experiments/phase8/ood/input_quality.py

Step 4: Lightweight Input Quality Analyzer for Multimodal Deepfake Forensics.
Extracts objective video and audio signal quality metrics without altering
the downstream classifier predictions.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
import torch


class InputQualityAnalyzer:
    """
    Lightweight, deterministic inspector of raw video and audio inputs.
    Extracts objective metrics to assess signal fidelity, ROI proportions,
    and compression anomalies for forensic reliability scoring.
    """

    MIN_FACE_AREA_RATIO = 0.15      # Faces occupying < 15% of frame area lack high-frequency forensic resolution
    SHARPNESS_DEFOCUS_THRESH = 45.0 # Laplacian variance < 45 indicates severe defocus blur
    AUDIO_SILENCE_RMS_THRESH = 1e-4 # RMS energy below 1e-4 indicates silent track
    AUDIO_CLIPPING_THRESH = 0.99    # Peak amplitude threshold for clipping detection

    def __init__(self):
        pass

    def analyze_video_properties(
        self,
        video_path: Path,
        raw_frames: Optional[List[np.ndarray]] = None,
        detected_face_boxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
    ) -> Dict[str, Any]:
        """
        Extract video stream metadata, geometry, face coverage, and spatial quality metrics.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "readable": False,
                "error": f"Failed to open video container: {video_path.name}",
                "resolution": "Unknown",
                "aspect_ratio": 1.0,
                "face_detected": False,
                "usable_frames_pct": 0.0,
            }

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (total_frames / max(1.0, fps)) if fps > 0 else 0.0
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip() if fourcc_int > 0 else "h264"
        cap.release()

        aspect_ratio = round(w / max(1, h), 2)
        is_widescreen_desktop = aspect_ratio > 1.80 or (w >= 1280 and aspect_ratio > 1.6)

        # Process frames for sharpness, compression artifacts, and face coverage
        sharpness_values = []
        face_areas = []
        usable_frames_count = 0

        if raw_frames is not None and len(raw_frames) > 0:
            for idx, frame in enumerate(raw_frames):
                # 1. Laplacian variance for sharpness
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
                var_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpness_values.append(var_lap)

                # 2. Face ROI analysis
                bbox = detected_face_boxes[idx] if (detected_face_boxes and idx < len(detected_face_boxes)) else None
                if bbox is not None:
                    bw = max(0, bbox[2] - bbox[0])
                    bh = max(0, bbox[3] - bbox[1])
                    area_ratio = (bw * bh) / max(1, w * h)
                    face_areas.append(area_ratio)
                    usable_frames_count += 1

        avg_sharpness = round(float(np.mean(sharpness_values)), 2) if sharpness_values else 100.0
        avg_face_area = round(float(np.mean(face_areas)), 4) if face_areas else 0.0
        usable_pct = round((usable_frames_count / max(1, len(raw_frames or []))) * 100.0, 1) if raw_frames else 100.0

        # Quality warnings
        quality_warnings = []
        if avg_face_area > 0 and avg_face_area < self.MIN_FACE_AREA_RATIO:
            quality_warnings.append(
                f"Small facial ROI ({avg_face_area*100:.1f}% of frame area < {self.MIN_FACE_AREA_RATIO*100:.0f}% minimum benchmark standard)."
            )
        if is_widescreen_desktop:
            quality_warnings.append(
                f"Widescreen desktop geometry ({w}x{h}, aspect ratio {aspect_ratio:.2f}) indicates potential screen recording."
            )
        if avg_sharpness < self.SHARPNESS_DEFOCUS_THRESH:
            quality_warnings.append(
                f"Low frame sharpness ({avg_sharpness:.1f} < {self.SHARPNESS_DEFOCUS_THRESH:.0f} threshold) indicates severe optical defocus or blur."
            )

        return {
            "readable": True,
            "width": w,
            "height": h,
            "resolution": f"{w}x{h}",
            "aspect_ratio": aspect_ratio,
            "is_widescreen_desktop": is_widescreen_desktop,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "codec": codec_str or "h264",
            "frame_sharpness": avg_sharpness,
            "face_detected": usable_frames_count > 0 or (detected_face_boxes is not None and any(b is not None for b in detected_face_boxes)),
            "face_area_ratio": avg_face_area,
            "usable_frames_pct": usable_pct,
            "quality_warnings": quality_warnings,
        }

    def analyze_audio_properties(
        self,
        waveform: Optional[torch.Tensor],
        sample_rate: int = 16000,
        audio_available: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract audio stream presence, RMS energy, clipping, and dynamic range metrics.
        """
        if not audio_available or waveform is None or waveform.numel() == 0:
            return {
                "audio_exists": False,
                "audio_readable": False,
                "duration_sec": 0.0,
                "sample_rate": sample_rate,
                "channels": 0,
                "is_silent": True,
                "rms_energy": 0.0,
                "peak_amplitude": 0.0,
                "clipping_ratio": 0.0,
                "snr_estimate_db": 0.0,
                "quality_warnings": ["Audio stream is absent, silent, or unreadable; vision-only fallback activated."],
            }

        # Convert to 1D float numpy array
        sig = waveform.squeeze().cpu().numpy().astype(np.float32)
        num_samples = len(sig)
        duration = num_samples / max(1, sample_rate)

        # 1. Peak & RMS energy
        peak = float(np.max(np.abs(sig)))
        rms = float(np.sqrt(np.mean(sig ** 2)))
        is_silent = rms < self.AUDIO_SILENCE_RMS_THRESH or peak < 1e-4

        # 2. Clipping detection
        clipped_samples = int(np.sum(np.abs(sig) >= self.AUDIO_CLIPPING_THRESH))
        clipping_ratio = round(clipped_samples / max(1, num_samples), 4)

        # 3. Estimated SNR (Signal-to-Noise Ratio)
        # Using 10th percentile energy as noise floor estimate
        frame_len = min(512, num_samples)
        if num_samples > frame_len * 4 and not is_silent:
            num_frames = num_samples // frame_len
            frame_energies = np.array([
                np.mean(sig[i*frame_len : (i+1)*frame_len]**2) for i in range(num_frames)
            ])
            noise_floor = max(1e-9, float(np.percentile(frame_energies, 10)))
            signal_power = max(1e-9, float(np.mean(frame_energies)))
            snr_db = round(10.0 * np.log10(signal_power / noise_floor), 1)
        else:
            snr_db = 0.0

        quality_warnings = []
        if is_silent:
            quality_warnings.append("Audio waveform contains near-zero RMS energy (silent acoustic stream).")
        if clipping_ratio > 0.02:
            quality_warnings.append(f"Acoustic waveform displays severe digital clipping ({clipping_ratio*100:.1f}% samples).")
        if sample_rate != 16000:
            quality_warnings.append(f"Native audio sample rate ({sample_rate} Hz) differs from benchmark 16,000 Hz domain.")

        return {
            "audio_exists": True,
            "audio_readable": True,
            "duration_sec": round(duration, 2),
            "sample_rate": sample_rate,
            "channels": 1 if waveform.ndim <= 2 else waveform.shape[0],
            "is_silent": is_silent,
            "rms_energy": round(rms, 6),
            "peak_amplitude": round(peak, 4),
            "clipping_ratio": clipping_ratio,
            "snr_estimate_db": snr_db,
            "quality_warnings": quality_warnings,
        }

    def evaluate_overall_quality(
        self,
        video_props: Dict[str, Any],
        audio_props: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesize video and audio quality profiles into an objective input quality assessment.
        """
        all_warnings = video_props.get("quality_warnings", []) + audio_props.get("quality_warnings", [])
        
        # Calculate quality penalty score
        penalty = 0.0
        if video_props.get("face_area_ratio", 0.5) < self.MIN_FACE_AREA_RATIO:
            penalty += 0.35
        if video_props.get("is_widescreen_desktop", False):
            penalty += 0.30
        if video_props.get("frame_sharpness", 100.0) < self.SHARPNESS_DEFOCUS_THRESH:
            penalty += 0.20
        if audio_props.get("is_silent", False) or not audio_props.get("audio_exists", True):
            penalty += 0.15

        overall_score = max(0.1, min(1.0, round(1.0 - penalty, 2)))

        if overall_score >= 0.80 and len(all_warnings) == 0:
            quality_grade = "OPTIMAL"
            summary = "Input quality meets high-fidelity forensic standards across spatial and acoustic streams."
        elif overall_score >= 0.55:
            quality_grade = "ACCEPTABLE"
            summary = "Input quality is sufficient for forensic evaluation with minor stream variances noted."
        else:
            quality_grade = "DEGRADED"
            summary = "Input quality is severely compromised; physical artifacts or out-of-domain geometry detected."

        return {
            "quality_grade": quality_grade,
            "quality_score": overall_score,
            "video": video_props,
            "audio": audio_props,
            "all_warnings": all_warnings,
            "summary": summary,
        }
