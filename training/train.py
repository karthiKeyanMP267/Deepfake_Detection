import os
import random
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from models.multimodal_model import MultimodalDeepfakeDetector
from training.dataloader import (
    get_train_loader,
    get_val_loader,
)
from training.losses import MultimodalLoss
from training.metrics import MetricTracker


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = -1.0
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_f1):
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")


class Trainer:

    def __init__(
        self,
        train_csv,
        val_csv,
        batch_size=8,
        epochs=15,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device=None,
        save_dir=None,
        experiment_name=None,
        subsample_ratio=None,
        use_sampler=True,
        use_class_weights=False,
        test_csv=None,
        enable_video_swin=True,
        enable_wav2vec2=False,
        freeze_strategy="full",
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.epochs = epochs
        self.use_sampler = use_sampler
        self.use_class_weights = use_class_weights
        self.enable_video_swin = enable_video_swin
        self.enable_wav2vec2 = enable_wav2vec2
        self.freeze_strategy = freeze_strategy

        # Resolve target experiment name and directories
        if experiment_name is not None:
            self.experiment_name = experiment_name
            self.base_dir = os.path.join("experiments", experiment_name)
        elif save_dir is not None:
            self.base_dir = save_dir
            self.experiment_name = os.path.basename(save_dir)
        else:
            self.experiment_name = "baseline"
            self.base_dir = os.path.join("experiments", "baseline")

        # Define subdirectories
        self.models_dir = os.path.join(self.base_dir, "models")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        self.metrics_dir = os.path.join(self.base_dir, "metrics")
        self.predictions_dir = os.path.join(self.base_dir, "predictions")
        self.reports_dir = os.path.join(self.base_dir, "reports")

        # Create all subdirectories
        for d in [self.models_dir, self.logs_dir, self.plots_dir, self.metrics_dir, self.predictions_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)

        self.save_path = os.path.join(self.models_dir, "best_model.pth")

        # -----------------------------
        # Dataloaders
        # -----------------------------

        self.train_loader = get_train_loader(
            train_csv,
            batch_size=batch_size,
            subsample_ratio=subsample_ratio,
            use_sampler=use_sampler,
        )

        self.val_loader = get_val_loader(
            val_csv,
            batch_size=batch_size,
            subsample_ratio=subsample_ratio,
        )

        # Calculate sample counts
        from collections import Counter
        train_labels = self.train_loader.dataset.data["label"].tolist()
        class_counts = Counter(train_labels)
        total_train_samples = len(train_labels)
        val_samples = len(self.val_loader.dataset)

        test_samples = 0
        if test_csv and os.path.exists(test_csv):
            test_df = pd.read_csv(test_csv)
            if subsample_ratio is not None and subsample_ratio < 1.0:
                test_df = test_df.sample(frac=subsample_ratio, random_state=42)
            test_samples = len(test_df)

        # -----------------------------
        # Model
        # -----------------------------

        self.model = MultimodalDeepfakeDetector(
            enable_efficientnet=True,
            enable_video_swin=enable_video_swin,
            enable_whisper=True,
            enable_wav2vec2=enable_wav2vec2,
            freeze_backbones=(freeze_strategy == "full"),
            freeze_strategy=freeze_strategy,
        )

        self.model.to(self.device)

        # -----------------------------
        # Loss & Class Weights
        # -----------------------------
        self.computed_class_weights = None
        if use_class_weights:
            num_classes = len(class_counts)
            weights = [total_train_samples / (num_classes * class_counts[i]) for i in sorted(class_counts.keys())]
            self.computed_class_weights = weights
            class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(self.device)
            self.criterion = MultimodalLoss(class_weights=class_weights_tensor)
        else:
            self.criterion = MultimodalLoss()

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine active vision and audio models from self.model
        vision_model_str = "EfficientNet-B0"
        if hasattr(self.model, "enable_video_swin") and self.model.enable_video_swin:
            if hasattr(self.model, "enable_efficientnet") and self.model.enable_efficientnet:
                vision_model_str = "EfficientNet-B0 + Video Swin Transformer"
            else:
                vision_model_str = "Video Swin Transformer"

        audio_model_str = "Whisper"
        if hasattr(self.model, "enable_wav2vec2") and self.model.enable_wav2vec2:
            if hasattr(self.model, "enable_whisper") and self.model.enable_whisper:
                audio_model_str = "Whisper + Wav2Vec2"
            else:
                audio_model_str = "Wav2Vec2"

        # Display Pre-Training Verification
        print("=" * 52)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 52)
        print(f"Experiment Name          : {self.experiment_name}")
        print(f"Architecture             : Multimodal")
        print(f"Vision                   : {vision_model_str}")
        print(f"Audio                    : {audio_model_str}")
        print(f"Freezing Strategy        : {freeze_strategy}")
        print(f"Fusion                   : Adaptive Attention Fusion")
        print(f"Contrastive              : Enabled")
        print(f"Training Sampler         : {'Enabled' if use_sampler else 'Disabled'}")
        print(f"Shuffle                  : Enabled")
        print(f"Loss Function            : {'Class-Weighted CrossEntropyLoss' if use_class_weights else 'Standard / Focal Loss'}")
        print(f"Computed Class Weights   : {self.computed_class_weights}")
        print(f"Optimizer                : AdamW")
        print(f"Learning Rate            : {learning_rate}")
        print(f"Weight Decay             : {weight_decay}")
        print(f"Scheduler                : ReduceLROnPlateau")
        print(f"Batch Size               : {batch_size}")
        print(f"Epochs                   : {epochs}")
        print(f"Random Seed              : 42")
        print(f"Training Samples         : {total_train_samples}")
        print(f"Validation Samples       : {val_samples}")
        print(f"Testing Samples          : {test_samples}")
        print(f"Checkpoint Save Directory: {self.models_dir}")
        print("=" * 52)

        # Save complete experiment configuration (config.json)
        config = {
            "Experiment Name": self.experiment_name,
            "Timestamp": timestamp,
            "Architecture": "Multimodal",
            "Vision Model": vision_model_str,
            "Audio Model": audio_model_str,
            "Freezing Strategy": freeze_strategy,
            "Fusion Method": "Adaptive Attention Fusion",
            "Contrastive Enabled": True,
            "Loss Function": "Class-Weighted CrossEntropyLoss" if use_class_weights else "Focal Loss",
            "Computed Class Weights": self.computed_class_weights,
            "Sampler": "Enabled" if use_sampler else "Disabled",
            "Shuffle": "Enabled",
            "Optimizer": "AdamW",
            "Learning Rate": learning_rate,
            "Weight Decay": weight_decay,
            "Scheduler": "ReduceLROnPlateau",
            "Batch Size": batch_size,
            "Epochs": epochs,
            "Random Seed": 42,
            "Dataset Used": train_csv,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "device": str(self.device),
            "subsample_ratio": subsample_ratio,
            "Checkpoint Path": os.path.join(self.models_dir, "best_model.pth"),
            "Git Commit": "N/A"
        }
        with open(os.path.join(self.logs_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        with open(os.path.join(self.reports_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        with open(os.path.join(self.base_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # Initialize training log CSV
        self.log_csv_path = os.path.join(self.logs_dir, "training_log.csv")
        with open(self.log_csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss,train_accuracy,val_accuracy,precision,recall,f1,auc\n")
        with open(os.path.join(self.reports_dir, "training_log.csv"), "w") as f:
            f.write("epoch,train_loss,val_loss,train_accuracy,val_accuracy,precision,recall,f1,auc\n")

        # Initialize history dict
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
            "precision_real": [],
            "recall_real": [],
            "f1_real": [],
            "precision_fake": [],
            "recall_fake": [],
            "f1_fake": []
        }

        # -----------------------------
        # Optimizer
        # -----------------------------

        if freeze_strategy == "partial":
            backbone_modules = [
                self.model.vision_encoder,
                self.model.video_swin_encoder,
                self.model.audio_encoder,
                self.model.wav2vec2_encoder,
            ]
            backbone_param_ids = set()
            for b in backbone_modules:
                if b is not None:
                    for p in b.parameters():
                        backbone_param_ids.add(id(p))

            group1_params = []
            group2_params = []
            for p in self.model.parameters():
                if p.requires_grad:
                    if id(p) in backbone_param_ids:
                        group2_params.append(p)
                    else:
                        group1_params.append(p)

            self.optimizer = optim.AdamW([
                {"params": group1_params, "lr": learning_rate, "weight_decay": weight_decay},
                {"params": group2_params, "lr": 1e-5, "weight_decay": weight_decay},
            ])
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        # -----------------------------
        # Scheduler
        # -----------------------------

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

        # Automatic Mixed Precision
        self.use_amp = ("cuda" in str(self.device))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.best_f1 = -1.0
        self.best_metrics = None
        self.save_dir = self.base_dir

    # ======================================================
    # TRAIN ONE EPOCH
    # ======================================================

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        tracker = MetricTracker()

        progress = tqdm(
            self.train_loader,
            desc="Training",
        )

        for batch in progress:

            self.optimizer.zero_grad()

            images = batch["images"].to(self.device)

            waveform = batch["waveform"].to(self.device)

            labels = batch["label"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    images,
                    waveform,
                )

                losses = self.criterion(
                    outputs,
                    labels,
                )

            if self.use_amp:
                self.scaler.scale(losses["loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["loss"].backward()
                self.optimizer.step()

            running_loss += losses["loss"].item()

            tracker.update(
                outputs["logits"],
                labels,
            )

        metrics = tracker.compute()

        metrics["loss"] = running_loss / len(self.train_loader)

        return metrics

    # ======================================================
    # VALIDATION
    # ======================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        tracker = MetricTracker()

        progress = tqdm(
            self.val_loader,
            desc="Validation",
        )

        for batch in progress:

            images = batch["images"].to(self.device)

            waveform = batch["waveform"].to(self.device)

            labels = batch["label"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    images,
                    waveform,
                )

                losses = self.criterion(
                    outputs,
                    labels,
                )

            running_loss += losses["loss"].item()

            tracker.update(
                outputs["logits"],
                labels,
            )

        metrics = tracker.compute()

        metrics["loss"] = running_loss / len(self.val_loader)

        return metrics

    # ======================================================
    # SAVE CHECKPOINT
    # ======================================================

    def save_checkpoint(self, epoch, metrics, filename="best_model.pth"):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),

            "scheduler_state_dict": self.scheduler.state_dict(),

            "best_f1": self.best_f1,

            "best_accuracy": metrics.get("accuracy", 0.0),

        }

        save_path = os.path.join(self.models_dir, filename)
        torch.save(
            checkpoint,
            save_path,
        )

    # ======================================================
    # TRAINING LOOP
    # ======================================================

    def fit(self):

        print("=" * 70)
        print("Starting Training")
        print("=" * 70)

        val_metrics = None
        epoch = 0
        early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

        try:
            for epoch in range(1, self.epochs + 1):

                print(f"\nEpoch {epoch}/{self.epochs}")

                train_metrics = self.train_one_epoch()

                val_metrics = self.validate()

                self.scheduler.step(
                    val_metrics["f1"]
                )

                print("-" * 70)

                print(
                    f"Train Loss : {train_metrics['loss']:.4f}"
                )

                print(
                    f"Train Accuracy : {train_metrics['accuracy']:.4f}"
                )

                print(
                    f"Validation Loss : {val_metrics['loss']:.4f}"
                )

                print(
                    f"Validation Accuracy : {val_metrics['accuracy']:.4f}"
                )

                print(
                    f"Validation Precision : {val_metrics['precision']:.4f}"
                )

                print(
                    f"Validation Recall : {val_metrics['recall']:.4f}"
                )

                print(
                    f"Validation F1 : {val_metrics['f1']:.4f}"
                )

                print(
                    f"Validation AUC : {val_metrics['auc']:.4f}"
                )

                print("-" * 70)

                # Append to history
                self.history["train_loss"].append(train_metrics["loss"])
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["train_accuracy"].append(train_metrics["accuracy"])
                self.history["val_accuracy"].append(val_metrics["accuracy"])
                self.history["precision"].append(val_metrics["precision"])
                self.history["recall"].append(val_metrics["recall"])
                self.history["f1"].append(val_metrics["f1"])
                self.history["auc"].append(val_metrics["auc"])
                
                # Per-class F1 metrics
                self.history["precision_real"].append(val_metrics["precision_real"])
                self.history["recall_real"].append(val_metrics["recall_real"])
                self.history["f1_real"].append(val_metrics["f1_real"])
                self.history["precision_fake"].append(val_metrics["precision_fake"])
                self.history["recall_fake"].append(val_metrics["recall_fake"])
                self.history["f1_fake"].append(val_metrics["f1_fake"])

                # Log to CSV
                with open(self.log_csv_path, "a") as f:
                    f.write(
                        f"{epoch},{train_metrics['loss']:.4f},{val_metrics['loss']:.4f},"
                        f"{train_metrics['accuracy']:.4f},{val_metrics['accuracy']:.4f},"
                        f"{val_metrics['precision']:.4f},{val_metrics['recall']:.4f},"
                        f"{val_metrics['f1']:.4f},{val_metrics['auc']:.4f}\n"
                    )

                if val_metrics["f1"] > self.best_f1:

                    self.best_f1 = val_metrics["f1"]
                    self.best_metrics = val_metrics

                    self.save_checkpoint(
                        epoch,
                        val_metrics,
                    )

                    # Save standard and normalized confusion matrix as .npy
                    np.save(
                        os.path.join(self.metrics_dir, "confusion_matrix.npy"),
                        val_metrics["confusion_matrix"]
                    )
                    np.save(
                        os.path.join(self.metrics_dir, "confusion_matrix_normalized.npy"),
                        val_metrics["confusion_matrix_normalized"]
                    )

                    print(
                        f"Best model saved "
                        f"(F1 = {self.best_f1:.4f})"
                    )

                # Early stopping check
                early_stopping(val_metrics["f1"])
                if early_stopping.early_stop:
                    break

            print("\nTraining Completed.")
            if val_metrics is not None:
                self.save_checkpoint(epoch, val_metrics, filename="final_model.pth")
                print("Final model saved (final_model.pth)")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current model checkpoint...")
            temp_metrics = val_metrics if val_metrics is not None else {"accuracy": 0.0}
            self.save_checkpoint(epoch, temp_metrics, filename="interrupted_model.pth")
            print("Interrupted model checkpoint saved (interrupted_model.pth)")
            raise KeyboardInterrupt

        # Save metrics.json
        if self.best_metrics is not None:
            serializable_metrics = {
                k: float(v) if isinstance(v, (np.float32, np.float64, float)) else v
                for k, v in self.best_metrics.items()
                if k not in ["confusion_matrix", "confusion_matrix_normalized"]
            }
            with open(os.path.join(self.metrics_dir, "metrics.json"), "w") as f:
                json.dump(serializable_metrics, f, indent=4)
            with open(os.path.join(self.base_dir, "metrics.json"), "w") as f:
                json.dump(serializable_metrics, f, indent=4)

        # Save history.json and training_history.json
        with open(os.path.join(self.logs_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=4)
        with open(os.path.join(self.logs_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=4)
        with open(os.path.join(self.base_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=4)

        # Plot training curves
        epochs_range = range(1, len(self.history["train_loss"]) + 1)

        # Loss curve
        plt.figure()
        plt.plot(epochs_range, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs_range, self.history["val_loss"], label="Val Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "loss_curve.png"))
        plt.savefig(os.path.join(self.base_dir, "loss_curve.png"))
        plt.close()

        # Accuracy curve
        plt.figure()
        plt.plot(epochs_range, self.history["train_accuracy"], label="Train Accuracy")
        plt.plot(epochs_range, self.history["val_accuracy"], label="Val Accuracy")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "accuracy_curve.png"))
        plt.savefig(os.path.join(self.base_dir, "accuracy_curve.png"))
        plt.close()

        # Combined training curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(epochs_range, self.history["train_loss"], label="Train Loss")
        axes[0].plot(epochs_range, self.history["val_loss"], label="Val Loss")
        axes[0].set_title("Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(epochs_range, self.history["train_accuracy"], label="Train Accuracy")
        axes[1].plot(epochs_range, self.history["val_accuracy"], label="Val Accuracy")
        axes[1].set_title("Accuracy Curve")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "training_curves.png"))
        plt.savefig(os.path.join(self.base_dir, "training_curves.png"))
        plt.close()

        # F1 Curve
        plt.figure()
        plt.plot(epochs_range, self.history["f1"], label="Val F1")
        plt.plot(epochs_range, self.history["f1_real"], label="Val F1 Real")
        plt.plot(epochs_range, self.history["f1_fake"], label="Val F1 Fake")
        plt.title("F1 Curve")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "f1_curve.png"))
        plt.close()

        print(
            f"Best Validation F1-score : "
            f"{self.best_f1:.4f}"
        )

        # ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    set_seed(42)

    # ------------------------------------------------------
    # Dataset CSV Files
    # ------------------------------------------------------

    TRAIN_CSV = "processed/train.csv"

    VAL_CSV = "processed/val.csv"

    # ------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------

    BATCH_SIZE = 2

    EPOCHS = 15

    LEARNING_RATE = 1e-4

    WEIGHT_DECAY = 1e-5

    SUBSAMPLE_RATIO = 0.25

    EXPERIMENT_NAME = "focal_loss"

    # ------------------------------------------------------
    # Trainer
    # ------------------------------------------------------

    trainer = Trainer(

        train_csv=TRAIN_CSV,

        val_csv=VAL_CSV,

        batch_size=BATCH_SIZE,

        epochs=EPOCHS,

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        subsample_ratio=SUBSAMPLE_RATIO,

        experiment_name=EXPERIMENT_NAME,

    )

    # ------------------------------------------------------
    # Start Training
    # ------------------------------------------------------

    trainer.fit()