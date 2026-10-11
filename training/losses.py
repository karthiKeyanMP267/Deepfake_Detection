import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.
    Standard formulation: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha weight dynamically based on targets:
        # Target = 1 (Fake) gets self.alpha
        # Target = 0 (Real) gets 1 - self.alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==========================================================
# MULTIMODAL LOSS
# ==========================================================

class MultimodalLoss(nn.Module):

    def __init__(
        self,
        classification_weight=1.0,
        contrastive_weight=0.2,
        class_weights=None,
    ):

        super().__init__()

        if class_weights is not None:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.classification_loss = FocalLoss(alpha=0.25, gamma=2.0)

        self.classification_weight = classification_weight

        self.contrastive_weight = contrastive_weight

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        outputs,
        labels,
    ):

        logits = outputs["logits"]

        contrastive_loss = outputs["contrastive_loss"]

        cls_loss = self.classification_loss(
            logits,
            labels,
        )

        total_loss = (

            self.classification_weight * cls_loss

            +

            self.contrastive_weight * contrastive_loss

        )

        losses = {

            "loss": total_loss,

            "classification_loss": cls_loss,

            "contrastive_loss": contrastive_loss,

        }

        return losses