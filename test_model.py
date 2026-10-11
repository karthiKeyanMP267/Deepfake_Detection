import torch

from training.dataloader import get_train_loader
from models.multimodal_model import MultimodalDeepfakeDetector

# Load one batch
loader = get_train_loader(
    "processed/train.csv",
    batch_size=2,
    num_workers=0,
)

batch = next(iter(loader))

images = batch["images"]
waveform = batch["waveform"]

# Create model
model = MultimodalDeepfakeDetector()

# Forward pass
with torch.no_grad():
    outputs = model(images, waveform)

print("\nOutput Keys:")
print(outputs.keys())

print("\nShapes:")
print("Logits             :", outputs["logits"].shape)
print("Vision Embedding   :", outputs["vision_embedding"].shape)
print("Audio Embedding    :", outputs["audio_embedding"].shape)
print("Fused Embedding    :", outputs["fused_embedding"].shape)
print("Attention          :", outputs["attention"].shape)

print("\nContrastive Loss:")
print(outputs["contrastive_loss"])

print("\nSimilarity Matrix:")
print(outputs["similarity"].shape)