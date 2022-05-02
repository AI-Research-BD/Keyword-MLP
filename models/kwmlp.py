"""Keyword-MLP model."""

from torch import nn
from einops.layers.torch import Rearrange, Reduce
from models.blocks import PreNorm, PostNorm, Residual, gMLPBlock, dropout_layers
from typing import Tuple


class KW_MLP(nn.Module):
    """Keyword-MLP."""

    def __init__(
        self,
        input_res: Tuple[int, int] = [40, 98],
        num_classes: int = 35,
        dim: int = 64,
        proj_mult: int = 4,
        depth: int = 12,
        prob_survival: float = 0.9,
        pre_norm: bool = False,
        **kwargs
    ) -> None:
        """Keyword-MLP for Speech Commands KWS task.

        Args:
            input_res (Tuple[int, int], optional): Resolution of input MFCC or Spectrogram, in the form [timesteps, mel_bins]. Defaults to [98, 40].
            num_classes (int, optional): Number of classes. Defaults to 35.
            dim (int, optional): Frequency embedding size. Defaults to 64.
            proj_mult (int, optional): Projection scale along frequency axis. Defaults to 4.
            depth (int, optional): Number of consecutive gMLP blocks. Defaults to 12.
            prob_survival (float, optional): Survival probability of each gMLP block. Defaults to 0.9.
            pre_norm (bool, optional): Whether to use pre-norm or post-norm. Defaults to False.
        """
        super().__init__()

        T, F = input_res
        dim_proj = dim * proj_mult
        Norm = PreNorm if pre_norm else PostNorm

        self.to_freq_embed = nn.Linear(F, dim)

        self.layers = nn.ModuleList(
            [
                Residual(Norm(dim, gMLPBlock(dim_f=dim, dim_f_proj=dim_proj, dim_t=T)))
                for _ in range(depth)
            ]
        )

        self.prob_survival = prob_survival

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce("b t d -> b d", "mean"),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.to_freq_embed(x)
        layers = (
            self.layers
            if not self.training
            else dropout_layers(self.layers, self.prob_survival)
        )
        x = nn.Sequential(*layers)(x)
        return self.to_logits(x)
