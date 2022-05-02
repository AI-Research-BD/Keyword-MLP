import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Cross Entropy with Label Smoothing.

    Attributes:
        num_classes (int): Number of target classes.
        smoothing (float, optional): Smoothing fraction constant, in the range (0.0, 1.0). Defaults to 0.1.
        dim (int, optional): Dimension across which to apply loss. Defaults to -1.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1, dim: int = -1):
        """Initializer for LabelSmoothingLoss.

        Args:
            num_classes (int): Number of target classes.
            smoothing (float, optional): Smoothing fraction constant, in the range (0.0, 1.0). Defaults to 0.1.
            dim (int, optional): Dimension across which to apply loss. Defaults to -1.
        """
        super().__init__()

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = num_classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): Model predictions, of shape (batch_size, num_classes).
            target (torch.Tensor): Target tensor of shape (batch_size).

        Returns:
            torch.Tensor: Loss.
        """
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class KDLoss(nn.Module):
    """Knowledge Distillation loss."""

    def __init__(self, num_classes: int, dim: int = -1):
        """Initializer for KDLoss.

        Args:
            num_classes (int): Number of target classes.
            dim (int, optional): Dimension across which to apply loss. Defaults to -1.
        """
        super().__init__()

        self.cls = num_classes
        self.dim = dim

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        teacher_pred: torch.Tensor,
        alpha: float,
        T: float,
    ) -> torch.Tensor:
        """Forward method for KDLoss.

        Args:
            pred (torch.Tensor): Predictions of student model. Tensor of shape (batch, num_classes).
            target (torch.Tensor): Labels. LongTensor of shape (batch,), containing class integers like [1, 2, 3, ...].
            teacher_pred (torch.Tensor): Predictions of teacher model. Tensor of shape (batch, num_classes).
            alpha (float): Weight parameter.
            T (float): Temperature for KDLoss.

        Returns:
            torch.Tensor: Loss value.
        """

        pred_log_probs = F.log_softmax(pred / T, dim=self.dim)
        teacher_pred_log_probs = F.log_softmax(teacher_pred / T, dim=self.dim)

        kldiv = F.kl_div(pred_log_probs, teacher_pred_log_probs, log_target=True)

        crossentropy = F.cross_entropy(pred, target)

        return (alpha * T * T) * kldiv + (1.0 - alpha) * crossentropy
