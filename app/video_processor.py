"""
app/video_processor.py

Video and Audio Preprocessing Pipeline for Live Inference.
Extracts normalized 16-frame face tensors and 16kHz mono audio waveforms.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image

import torch
import torchaudio
from torchvision import transforms

from app.config import (
    NUM_FRAMES,
    IMAGE_SIZE,
    AUDIO_SAMPLE_RATE,
    MAX_AUDIO_LENGTH_SEC,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
)
from preprocessing.audio_detector import AudioDetector


class VideoProcessor:
    """
    Handles uploaded video ingestion, face extraction, frame normalization,
    audio stream extraction, and audio availability verification.
    """

    def __init__(self):
        self.audio_detector = AudioDetector(sample_rate=AUDIO_SAMPLE_RATE)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self._face_detector = None

    def _get_face_detector(self):
        """Lazy load InsightFace or OpenCV face detector."""
        if self._face_detector is None:
            try:
                from insightface.app import FaceAnalysis
                from preprocessing.config import INSIGHTFACE_ROOT
                app = FaceAnalysis(
                    name="buffalo_l",
                    root=INSIGHTFACE_ROOT,
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=-1, det_size=(320, 320))
                self._face_detector = app
            except Exception:
                self._face_detector = "fallback_crop"
        return self._face_detector

    def validate_file(self, file_path: Path, file_size: int) -> Tuple[bool, str]:
        """
        Validate file extension and file size.
        """
        if file_size > MAX_UPLOAD_BYTES:
            return False, f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed size ({MAX_UPLOAD_BYTES / (1024*1024)} MB)."

        ext = file_path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

        return True, "File validation passed."

    def extract_frames(self, video_path: Path, num_frames: int = NUM_FRAMES) -> List[np.ndarray]:
        """
        Extract uniformly sampled RGB frames from the video file.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file '{video_path.name}'. File may be corrupted or unreadable.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains 0 readable frames.")

        # Uniformly sample frame indices
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        frames = []
        current_idx = 0
        target_idx_set = set(indices)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in target_idx_set:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
            current_idx += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError("Could not decode any valid video frames.")

        # Pad if fewer than num_frames
        while len(frames) < num_frames:
            frames.append(frames[-1])

        return frames[:num_frames]

    def extract_face_crop(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Crop face region with padding. If no bbox, perform central portrait crop.
        """
        h, w = frame.shape[:2]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            padx, pady = int(bw * 0.15), int(bh * 0.15)
            x1 = max(0, x1 - padx)
            y1 = max(0, y1 - pady)
            x2 = min(w, x2 + padx)
            y2 = min(h, y2 + pady)
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0 and cropped.shape[0] > 10 and cropped.shape[1] > 10:
                return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))

        # Fallback: central square crop
        min_dim = min(h, w)
        sy = (h - min_dim) // 2
        sx = (w - min_dim) // 2
        cropped = frame[sy:sy + min_dim, sx:sx + min_dim]
        return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))

    def detect_video_face_bbox(self, frames: List[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face bounding box from sample frames using InsightFace/SCRFD or Haar Cascade.
        """
        detector = self._get_face_detector()
        if detector == "fallback_crop":
            return None

        try:
            for frame in frames[::4]:  # Check every 4th frame
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                faces = detector.get(bgr)
                if len(faces) > 0:
                    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    x1, y1, x2, y2 = largest.bbox.astype(int)
                    return (x1, y1, x2, y2)
        except Exception:
            pass

        return None

    def process_video_frames(self, video_path: Path) -> torch.Tensor:
        """
        Extract 16 normalized face/frame tensors using per-frame face tracking with fallback.
        Returns shape (1, 16, 3, 224, 224).
        """
        raw_frames = self.extract_frames(video_path, num_frames=NUM_FRAMES)
        detector = self._get_face_detector()

        tensor_list = []
        last_valid_bbox = None

        for frame in raw_frames:
            bbox = None
            if detector != "fallback_crop":
                try:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    faces = detector.get(bgr)
                    if len(faces) > 0:
                        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        bbox = tuple(map(int, largest.bbox))
                        last_valid_bbox = bbox
                except Exception:
                    pass

            if bbox is None and last_valid_bbox is not None:
                bbox = last_valid_bbox

            cropped = self.extract_face_crop(frame, bbox=bbox)
            pil_img = Image.fromarray(cropped)
            norm_tensor = self.transform(pil_img)
            tensor_list.append(norm_tensor)

        # Stack to (16, 3, 224, 224) -> (1, 16, 3, 224, 224)
        video_tensor = torch.stack(tensor_list, dim=0).unsqueeze(0)
        return video_tensor

    def extract_audio_stream(self, video_path: Path, output_wav_path: Path) -> Tuple[torch.Tensor, bool]:
        """
        Extract audio stream to WAV file, validate via AudioDetector, and return (waveform, audio_available).
        Target: shape (1, 160000) (10 seconds @ 16kHz).
        """
        max_samples = AUDIO_SAMPLE_RATE * MAX_AUDIO_LENGTH_SEC
        zero_waveform = torch.zeros((1, max_samples), dtype=torch.float32)

        # 1. Locate ffmpeg executable
        ffmpeg_cmd = "ffmpeg"
        try:
            import imageio_ffmpeg
            ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

        # 2. Run extraction
        cmd = [
            ffmpeg_cmd,
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-y",
            str(output_wav_path),
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=30,
            )
        except Exception:
            # Failed to extract audio or video has no audio track
            return zero_waveform, False

        # 3. Check audio file validity
        is_valid, reason = self.audio_detector.check_audio_file(output_wav_path)
        if not is_valid:
            return zero_waveform, False

        # 4. Load waveform with torchaudio
        try:
            waveform, sr = torchaudio.load(output_wav_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)

            # Check waveform numbers
            wf_valid, wf_reason = self.audio_detector.check_waveform(waveform.squeeze(0), AUDIO_SAMPLE_RATE)
            if not wf_valid:
                return zero_waveform, False

            # Pad or trim to exact max_samples
            num_samples = waveform.shape[-1]
            if num_samples < max_samples:
                padding = max_samples - num_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                waveform = waveform[:, :max_samples]

            return waveform, True

        except Exception:
            return zero_waveform, False

    def process_video_and_audio(
        self,
        video_path: Path,
        temp_dir: Path,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, dict]:
        """
        End-to-end extraction from a video file.
        Returns:
            images_tensor: (1, 16, 3, 224, 224)
            waveform_tensor: (1, 160000)
            audio_available: bool
            video_metadata: dict containing resolution, aspect ratio, face area ratio, etc.
        """
        # 1. Probe video resolution, codec, and duration
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 224
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 224
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 30.0
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 16
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC)) if cap.isOpened() else 0
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]) if fourcc_int > 0 else "h264"
        cap.release()
        vid_duration = (frame_cnt / fps) if fps > 0 else 0.0

        # 2. Extract and process video frames
        raw_frames = self.extract_frames(video_path, num_frames=NUM_FRAMES)
        detector = self._get_face_detector()

        tensor_list = []
        last_valid_bbox = None
        face_areas = []
        sharpness_list = []

        for frame in raw_frames:
            # Measure frame sharpness (Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            sharpness_list.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            bbox = None
            if detector != "fallback_crop":
                try:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    faces = detector.get(bgr)
                    if len(faces) > 0:
                        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        bbox = tuple(map(int, largest.bbox))
                        last_valid_bbox = bbox
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        face_areas.append((bw * bh) / max(1, w * h))
                except Exception:
                    pass

            if bbox is None and last_valid_bbox is not None:
                bbox = last_valid_bbox

            cropped = self.extract_face_crop(frame, bbox=bbox)
            pil_img = Image.fromarray(cropped)
            norm_tensor = self.transform(pil_img)
            tensor_list.append(norm_tensor)

        images_tensor = torch.stack(tensor_list, dim=0).unsqueeze(0)

        # 3. Extract and process audio
        temp_wav_path = temp_dir / "extracted_audio.wav"
        waveform_tensor, audio_available = self.extract_audio_stream(video_path, temp_wav_path)

        avg_face_ratio = float(np.mean(face_areas)) if len(face_areas) > 0 else 0.5
        avg_sharpness = float(np.mean(sharpness_list)) if len(sharpness_list) > 0 else 100.0

        metadata = {
            "width": w,
            "height": h,
            "resolution": f"{w}x{h}",
            "aspect_ratio": round(w / max(1, h), 2),
            "fps": round(fps, 2),
            "total_frames": frame_cnt,
            "duration_sec": round(vid_duration, 2),
            "codec": fourcc_str.strip() or "h264",
            "face_area_ratio": round(avg_face_ratio, 4),
            "frame_sharpness": round(avg_sharpness, 2),
            "audio_available": audio_available,
        }

        return images_tensor, waveform_tensor, audio_available, metadata
