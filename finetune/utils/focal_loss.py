import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossBCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossBCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (batch_size, *) raw model outputs (logits)
        # targets: (batch_size, *) binary labels (0, 1)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = sigmoid(inputs) if targets == 1 else 1 - sigmoid(inputs)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class FocalLossCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (batch_size, num_classes) raw model outputs (logits)
        # targets: (batch_size) class labels

        # Convert labels to one-hot encoding
        targets = F.one_hot(targets, num_classes=inputs.size(1))

        # Calculate cross entropy loss
        logpt = F.log_softmax(inputs, dim=1)  # compute log softmax along class dimension
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)  # pick the log probability for the correct class
        pt = logpt.exp()  # convert log probability to probability

        # Compute focal loss
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
