from torch import nn
from einops.layers.torch import Rearrange, Reduce
from models.blocks import PreNorm, PostNorm, Residual, TemporalGatingUnit, gMLPBlock, dropout_layers
from typing import Tuple



class KW_MLP(nn.Module):
    """Keyword-MLP."""
    
    def __init__(
        self,
        input_res: Tuple[int, int] = [40, 98],
        num_classes: int = 35,
        dim: int = 64,
        depth: int = 12,
        ff_mult: int = 4,
        prob_survival: float = 0.9,
        pre_norm: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.to_freq_embed = nn.Sequential(
            Rearrange('b f t -> b t f'),
            nn.Linear(input_res[0], dim)
        )
        
        Norm = PreNorm if pre_norm else PostNorm
        dim_ff = dim * ff_mult

        self.layers = nn.ModuleList(
            [Residual(Norm(dim, gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=input_res[1]))) for _ in range(depth)]
        )

        self.prob_survival = prob_survival
        
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b t d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_freq_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        return self.to_logits(x)