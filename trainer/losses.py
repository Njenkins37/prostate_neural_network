import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Computes the Dice loss. 
    Ignores background pixels and strictly evaluates the shape of the predicted tumor.
    """
    def __init__(self, smooth=1e-4):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to squash logits between 0 and 1.
        probs = torch.sigmoid(logits)
        
        # Flatten the tensors to 1D sequences.
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union.
        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1.0 - dice_score

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (Lin et al., 2017).
    Down-weights easy background pixels to focus gradients on hard tumor pixels.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Calculate standard BCE with logits (more numerically stable than sigmoid + BCE).
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate the modulating factor.
        # pt is the probability of the true class. 
        # Using math properties: pt = exp(-BCE).
        pt = torch.exp(-bce_loss) 
        
        # Focal Loss formula.
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

class CombinedFocalDiceLoss(nn.Module):
    """
    The ultimate medical segmentation loss function.
    Combines shape-awareness (Dice) with hard-pixel mining (Focal).
    """
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=1e-4)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        # Explicitly cast to 32-bit float to prevent probs.sum() 
        # from exceeding the 65,504 limit of FP16 memory.
        logits = logits.float()
        targets = targets.float()
        
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        
        return (self.dice_weight * dice) + (self.focal_weight * focal)