"""Composite CE + Dice + IoU loss with class weights, ported from the notebook."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, class_weights=None, epsilon=1e-12):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.epsilon = epsilon

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        numerator = 2.0 * torch.sum(probs * one_hot, dim=(-2, -1))
        denominator = torch.sum(probs + one_hot, dim=(-2, -1))
        loss = 1.0 - (numerator + self.epsilon) / (denominator + self.epsilon)
        if self.class_weights is not None:
            loss = loss * self.class_weights.reshape(1, num_classes)
        return loss.mean()


class IoULoss(nn.Module):
    def __init__(self, class_weights=None, smooth=1e-8):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.smooth = smooth

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * one_hot, dim=(2, 3))
        union = torch.sum(probs + one_hot, dim=(2, 3)) - intersection
        loss = 1.0 - (intersection + self.smooth) / (union + self.smooth)
        if self.class_weights is not None:
            loss = loss * self.class_weights.reshape(1, num_classes)
        return loss.mean()


class CombinedLoss(nn.Module):
    """ce * CrossEntropy + dice * SoftDice + iou * IoU, all class-weighted."""

    def __init__(self, ce=0.3, dice=0.35, iou=0.35, class_weights=None):
        super().__init__()
        self.weights = (ce, dice, iou)
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.soft_dice = SoftDiceLoss(class_weights)
        self.iou = IoULoss(class_weights)

    def forward(self, logits, target):
        # Explicit fp32, matching what autocast already does for softmax/sum.
        logits = logits.float()
        ce, dice, iou = self.weights
        return (ce * self.cross_entropy(logits, target)
                + dice * self.soft_dice(logits, target)
                + iou * self.iou(logits, target))
