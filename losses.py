import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0, reduction='mean'):
        """
        Focal Loss for imbalanced classification.
        alpha: Weight for the rare class (illicit). 
               Can be a single float or a tensor of shape (num_classes,).
        gamma: Focusing parameter to down-weight easy examples.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C] (logits)
        # targets: [N] (class indices)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply alpha weighting
        if isinstance(self.alpha, float):
            # If alpha is a single number, assume it's the weight for class 1
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss
        elif isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_weighted_ce_loss(targets):
    """
    Computes class weights inversely proportional to class frequencies 
    and returns a CrossEntropyLoss module.
    """
    class_counts = torch.bincount(targets)
    total_samples = len(targets)
    
    # Avoid division by zero by clamping
    class_counts = torch.clamp(class_counts, min=1)
    
    # Weight formula: N / (C * n_i)
    num_classes = len(class_counts)
    weights = total_samples / (num_classes * class_counts.float())
    
    return nn.CrossEntropyLoss(weight=weights)
