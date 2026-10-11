from training.dataloader import get_train_loader
from models.multimodal_model import MultimodalDeepfakeDetector
from training.losses import MultimodalLoss

loader = get_train_loader(
    "processed/train.csv",
    batch_size=2,
    num_workers=0,
)

batch = next(iter(loader))

model = MultimodalDeepfakeDetector()
criterion = MultimodalLoss()

outputs = model(
    batch["images"],
    batch["waveform"],
)

losses = criterion(
    outputs,
    batch["label"],
)

print(losses)