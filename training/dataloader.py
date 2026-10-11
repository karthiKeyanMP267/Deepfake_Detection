import os
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from PIL import Image

from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ==========================================================
# DATASET
# ==========================================================

class DeepfakeDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_size=224,
        sample_rate=16000,
        max_audio_length=10,
        transform=None,
        subsample_ratio=None,
    ):

        self.data = pd.read_csv(csv_file)

        if subsample_ratio is not None and subsample_ratio < 1.0:
            self.data = self.data.sample(frac=subsample_ratio, random_state=42).reset_index(drop=True)

        self.image_size = image_size
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length

        if transform is None:

            self.transform = transforms.Compose([

                transforms.Resize((image_size, image_size)),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),

            ])

        else:

            self.transform = transform

        from preprocessing.audio_detector import AudioDetector
        self.audio_detector = AudioDetector(sample_rate=self.sample_rate)

    # ======================================================
    # DATASET LENGTH
    # ======================================================

    def __len__(self):

        return len(self.data)

    # ======================================================
    # LOAD FACE IMAGES
    # ======================================================

    def load_images(self, folder):

        folder = Path(folder)

        image_files = sorted(folder.glob("*.jpg"))

        images = []

        for image_path in image_files:

            image = Image.open(image_path).convert("RGB")

            image = self.transform(image)

            images.append(image)

        images = torch.stack(images)

        return images

    # ======================================================
    # LOAD AUDIO
    # ======================================================

    def load_audio(self, audio_path):
        max_length = self.sample_rate * self.max_audio_length
        is_valid, reason = self.audio_detector.check_audio_file(audio_path)

        if not is_valid:
            waveform = torch.zeros(max_length, dtype=torch.float32)
            return waveform, False

        try:
            waveform, sr = torchaudio.load(audio_path)

            # Convert stereo → mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sr,
                    self.sample_rate
                )
                waveform = resampler(waveform)

            waveform = waveform.squeeze(0)

            # Pad audio
            if len(waveform) < max_length:
                padding = max_length - len(waveform)
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0, padding)
                )
            # Trim audio
            else:
                waveform = waveform[:max_length]

            # Re-check waveform validity after processing
            waveform_valid, wf_reason = self.audio_detector.check_waveform(waveform, self.sample_rate)
            if not waveform_valid:
                return torch.zeros(max_length, dtype=torch.float32), False

            return waveform, True

        except Exception:
            return torch.zeros(max_length, dtype=torch.float32), False

    # ======================================================
    # GET ITEM
    # ======================================================

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # --------------------------------------------------
        # Read CSV fields
        # --------------------------------------------------

        video_path = row["video_path"]

        audio_path = row["audio_path"]

        face_folder = row["face_folder"]

        label = int(row["label"])

        manipulation = row["manipulation"]

        # --------------------------------------------------
        # Load Images
        # --------------------------------------------------

        images = self.load_images(face_folder)

        # Ensure exactly 16 frames
        if images.shape[0] < 16:

            last_frame = images[-1]

            padding = last_frame.unsqueeze(0).repeat(
                16 - images.shape[0],
                1,
                1,
                1,
            )

            images = torch.cat(
                [images, padding],
                dim=0,
            )

        elif images.shape[0] > 16:

            images = images[:16]

        # --------------------------------------------------
        # Load Audio
        # --------------------------------------------------

        waveform, audio_available = self.load_audio(audio_path)

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        # --------------------------------------------------
        # Return Sample
        # --------------------------------------------------

        sample = {

            "images": images,

            "waveform": waveform,

            "audio_available": audio_available,

            "label": label,

            "manipulation": manipulation,

            "video_path": video_path,

            "audio_path": audio_path,

            "face_folder": face_folder,

        }

        return sample

# ==========================================================
# DATALOADER
# ==========================================================

def create_dataloader(
    csv_file,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    subsample_ratio=None,
    drop_last=False,
    use_sampler=True,
):

    dataset = DeepfakeDataset(csv_file, subsample_ratio=subsample_ratio)

    if shuffle:
        if use_sampler:
            labels = dataset.data["label"].tolist()
            class_counts = Counter(labels)
            sample_weights = [
                1.0 / class_counts[label]
                for label in labels
            ]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )
        else:
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                sampler=None,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )
    else:
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    return loader


# ==========================================================
# TRAIN LOADER
# ==========================================================

def get_train_loader(
    csv_file,
    batch_size=8,
    num_workers=4,
    subsample_ratio=None,
    use_sampler=True,
):

    return create_dataloader(
        csv_file=csv_file,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        subsample_ratio=subsample_ratio,
        drop_last=True,
        use_sampler=use_sampler,
    )


# ==========================================================
# VALIDATION LOADER
# ==========================================================

def get_val_loader(
    csv_file,
    batch_size=8,
    num_workers=4,
    subsample_ratio=None,
):

    return create_dataloader(
        csv_file=csv_file,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        subsample_ratio=subsample_ratio,
    )


# ==========================================================
# TEST LOADER
# ==========================================================

def get_test_loader(
    csv_file,
    batch_size=8,
    num_workers=4,
    subsample_ratio=None,
):

    return create_dataloader(
        csv_file=csv_file,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        subsample_ratio=subsample_ratio,
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    train_csv = "processed/train.csv"

    loader = get_train_loader(
        csv_file=train_csv,
        batch_size=2,
    )

    batch = next(iter(loader))

    print("=" * 60)

    print("Images Shape")
    print(batch["images"].shape)

    print()

    print("Waveform Shape")
    print(batch["waveform"].shape)

    print()

    print("Labels")
    print(batch["label"])

    print()

    print("Manipulation")
    print(batch["manipulation"])

    print("=" * 60)