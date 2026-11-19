import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, labels, reduction="none")
        probability = (-cross_entropy).exp()
        class_weight = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        return (class_weight * (1 - probability).pow(self.gamma) * cross_entropy).mean()


def get_weighted_ce_loss(labels: torch.Tensor) -> nn.CrossEntropyLoss:
    counts = torch.bincount(labels, minlength=2).clamp_min(1)
    weights = len(labels) / (2 * counts.float())
    return nn.CrossEntropyLoss(weight=weights)
