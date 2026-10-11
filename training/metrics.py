import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ==========================================================
# METRIC TRACKER
# ==========================================================

class MetricTracker:

    def __init__(self):

        self.reset()

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):

        self.labels = []

        self.predictions = []

        self.probabilities = []

    # ======================================================
    # UPDATE
    # ======================================================

    def update(
        self,
        logits,
        labels,
    ):

        probs = torch.softmax(
            logits,
            dim=1,
        )

        preds = torch.argmax(
            probs,
            dim=1,
        )

        self.labels.extend(
            labels.cpu().numpy().tolist()
        )

        self.predictions.extend(
            preds.cpu().numpy().tolist()
        )

        self.probabilities.extend(
            probs[:, 1].detach().cpu().numpy().tolist()
        )

    # ======================================================
    # COMPUTE
    # ======================================================

    def compute(self):

        labels = np.array(self.labels)

        preds = np.array(self.predictions)

        probs = np.array(self.probabilities)

        metrics = {}

        metrics["accuracy"] = accuracy_score(
            labels,
            preds,
        )

        metrics["precision"] = precision_score(
            labels,
            preds,
            zero_division=0,
        )

        metrics["recall"] = recall_score(
            labels,
            preds,
            zero_division=0,
        )

        metrics["f1"] = f1_score(
            labels,
            preds,
            zero_division=0,
        )

        try:

            metrics["auc"] = roc_auc_score(
                labels,
                probs,
            )

        except ValueError:

            metrics["auc"] = 0.0

        # Calculate per-class metrics
        metrics["precision_real"] = precision_score(
            labels,
            preds,
            pos_label=0,
            zero_division=0,
        )
        metrics["recall_real"] = recall_score(
            labels,
            preds,
            pos_label=0,
            zero_division=0,
        )
        metrics["f1_real"] = f1_score(
            labels,
            preds,
            pos_label=0,
            zero_division=0,
        )

        metrics["precision_fake"] = precision_score(
            labels,
            preds,
            pos_label=1,
            zero_division=0,
        )
        metrics["recall_fake"] = recall_score(
            labels,
            preds,
            pos_label=1,
            zero_division=0,
        )
        metrics["f1_fake"] = f1_score(
            labels,
            preds,
            pos_label=1,
            zero_division=0,
        )

        # Standard Confusion Matrix
        cm = confusion_matrix(
            labels,
            preds,
        )
        metrics["confusion_matrix"] = cm

        # Normalized Confusion Matrix
        row_sums = cm.sum(axis=1, keepdims=True)
        metrics["confusion_matrix_normalized"] = np.where(
            row_sums > 0,
            cm.astype('float') / row_sums,
            0.0
        )

        return metrics