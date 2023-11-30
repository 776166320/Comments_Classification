import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output: torch.tensor, target: torch.Tensor):
        """
        Args:
            pred: shape = [batch_size, num_classes]
            gt: shape = [batch_size]

        Returns:
            loss
        """

        loss = self.loss_fn(output, target)
        return loss