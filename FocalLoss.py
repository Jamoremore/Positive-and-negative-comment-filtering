import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
    def forward(self, inputs, targets):
        ce = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')
        logp = ce(inputs, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
        # loss = -1 * (1 - pt) ** self.gamma * logpt              # (batch, )