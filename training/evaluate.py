import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import torch

from models.multimodal_model import MultimodalDeepfakeDetector
from training.dataloader import get_test_loader
from training.metrics import MetricTracker


class Evaluator:

    def __init__(
        self,
        test_csv,
        checkpoint_path=None,
        experiment_name=None,
        batch_size=8,
        device=None,
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Resolve target experiment directories
        if experiment_name is not None:
            self.experiment_name = experiment_name
            self.base_dir = os.path.join("experiments", experiment_name)
        elif checkpoint_path is not None:
            parent = os.path.dirname(checkpoint_path)
            if os.path.basename(parent) == "models":
                self.base_dir = os.path.dirname(parent)
            else:
                self.base_dir = parent
            self.experiment_name = os.path.basename(self.base_dir)
        else:
            self.experiment_name = "baseline"
            self.base_dir = os.path.join("experiments", "baseline")

        # Define subdirectories
        self.models_dir = os.path.join(self.base_dir, "models")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        self.metrics_dir = os.path.join(self.base_dir, "metrics")
        self.predictions_dir = os.path.join(self.base_dir, "predictions")

        # Create all subdirectories
        for d in [self.models_dir, self.logs_dir, self.plots_dir, self.metrics_dir, self.predictions_dir]:
            os.makedirs(d, exist_ok=True)

        # Resolve checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.models_dir, "best_model.pth")

        self.checkpoint_path = checkpoint_path

        # -----------------------------
        # Test DataLoader
        # -----------------------------
        self.test_loader = get_test_loader(
            test_csv,
            batch_size=batch_size,
        )

        # -----------------------------
        # Model
        # -----------------------------
        self.model = MultimodalDeepfakeDetector()

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    # ======================================================
    # EVALUATE
    # ======================================================

    @torch.no_grad()
    def evaluate(self):

        tracker = MetricTracker()

        all_labels = []
        all_preds = []
        all_prob_real = []
        all_prob_fake = []

        for batch in self.test_loader:

            images = batch["images"].to(self.device)

            waveform = batch["waveform"].to(self.device)

            labels = batch["label"].to(self.device)

            outputs = self.model(
                images,
                waveform,
            )

            tracker.update(
                outputs["logits"],
                labels,
            )

            # Record predictions for predictions.csv
            probs = torch.softmax(outputs["logits"], dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_prob_real.extend(probs[:, 0].cpu().numpy().tolist())
            all_prob_fake.extend(probs[:, 1].cpu().numpy().tolist())

        metrics = tracker.compute()

        from sklearn.metrics import classification_report
        clf_report = classification_report(
            all_labels,
            all_preds,
            target_names=["Real", "Fake"],
            digits=4,
            zero_division=0
        )

        # Save predictions.csv
        predictions_df = pd.DataFrame({
            "ground_truth": all_labels,
            "prediction": all_preds,
            "probability_real": all_prob_real,
            "probability_fake": all_prob_fake
        })
        predictions_df.to_csv(
            os.path.join(self.predictions_dir, "predictions.csv"),
            index=False
        )
        predictions_df.to_csv(
            os.path.join(self.base_dir, "predictions.csv"),
            index=False
        )

        # Save metrics.json (test evaluation results)
        serializable_metrics = {
            k: float(v) if isinstance(v, (np.float32, np.float64, float)) else v
            for k, v in metrics.items()
            if k not in ["confusion_matrix", "confusion_matrix_normalized"]
        }
        with open(os.path.join(self.metrics_dir, "metrics.json"), "w") as f:
            json.dump(serializable_metrics, f, indent=4)
        with open(os.path.join(self.base_dir, "metrics.json"), "w") as f:
            json.dump(serializable_metrics, f, indent=4)

        # Save classification_report.txt
        with open(os.path.join(self.metrics_dir, "classification_report.txt"), "w") as f:
            f.write(clf_report)
        with open(os.path.join(self.base_dir, "classification_report.txt"), "w") as f:
            f.write(clf_report)

        # Plot and save curves
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_prob_fake)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['auc']:.4f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "roc_curve.png"))
        plt.savefig(os.path.join(self.plots_dir, "ROC.png"))
        plt.savefig(os.path.join(self.base_dir, "ROC.png"))
        plt.close()

        # 2. Precision-Recall Curve
        precisions, recalls, _ = precision_recall_curve(all_labels, all_prob_fake)
        plt.figure()
        plt.plot(recalls, precisions, label="PR Curve")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "precision_recall_curve.png"))
        plt.savefig(os.path.join(self.plots_dir, "PrecisionRecall.png"))
        plt.savefig(os.path.join(self.base_dir, "PrecisionRecall.png"))
        plt.close()

        # 3. Confusion Matrix Figure (Raw)
        plt.figure()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix"],
            display_labels=["Real", "Fake"]
        )
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.plots_dir, "confusion_matrix.png"))
        plt.savefig(os.path.join(self.base_dir, "confusion_matrix.png"))
        plt.close()

        # 4. Confusion Matrix Figure (Normalized)
        plt.figure()
        disp_norm = ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix_normalized"],
            display_labels=["Real", "Fake"]
        )
        disp_norm.plot(cmap=plt.cm.Blues, values_format=".4f")
        plt.title("Normalized Confusion Matrix")
        plt.savefig(os.path.join(self.plots_dir, "confusion_matrix_normalized.png"))
        plt.savefig(os.path.join(self.plots_dir, "normalized_confusion_matrix.png"))
        plt.savefig(os.path.join(self.base_dir, "normalized_confusion_matrix.png"))
        plt.close()

        return metrics

    # ======================================================
    # PRINT RESULTS
    # ======================================================

    def print_results(self, metrics):

        print("=" * 70)

        print("TEST RESULTS")

        print("=" * 70)

        print(f"Accuracy  : {metrics['accuracy']:.4f}")

        print(f"Precision : {metrics['precision']:.4f}")

        print(f"Recall    : {metrics['recall']:.4f}")

        print(f"F1 Score  : {metrics['f1']:.4f}")

        print(f"ROC-AUC   : {metrics['auc']:.4f}")

        print("-" * 50)
        print("PER-CLASS METRICS:")
        print(f"Real Class (Label 0) - Precision: {metrics['precision_real']:.4f}, Recall: {metrics['recall_real']:.4f}, F1: {metrics['f1_real']:.4f}")
        print(f"Fake Class (Label 1) - Precision: {metrics['precision_fake']:.4f}, Recall: {metrics['recall_fake']:.4f}, F1: {metrics['f1_fake']:.4f}")

        print("\nConfusion Matrix (Raw):")

        print(metrics["confusion_matrix"])

        print("\nConfusion Matrix (Normalized):")

        print(metrics["confusion_matrix_normalized"])

        print("=" * 70)


if __name__ == "__main__":

    TEST_CSV = "processed/test.csv"
    EXPERIMENT_NAME = "focal_loss"

    evaluator = Evaluator(
        test_csv=TEST_CSV,
        experiment_name=EXPERIMENT_NAME,
        batch_size=2,  # Reduce batch size to prevent OOM
    )

    metrics = evaluator.evaluate()
    evaluator.print_results(metrics)