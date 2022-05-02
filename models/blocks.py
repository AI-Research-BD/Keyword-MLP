###############################################################################
# The code for the kw-mlp model is mostly adapted from lucidrains/g-mlp-pytorch
###############################################################################
# MIT License
#
# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import torch
from torch import nn
import torch.nn.functional as F
from random import randrange
from typing import List


def dropout_layers(layers: List[nn.Module], prob_survival: float) -> List[nn.Module]:
    """Drops layers with a certain probability, keeping at least one layer.

    Args:
        layers (List[nn.Module]): List of layers.
        prob_survival (float): Survival probability of layers.

    Returns:
        List[nn.Module]: New list of layers.
    """

    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0.0, 1.0) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


class Residual(nn.Module):
    """Wrapper to implement shallow skip-connection."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    """Applies LayerNorm before target operation."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    """Applies LayerNorm after target operation."""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class TemporalGatingUnit(nn.Module):
    """Linear projection across temporal axis and linear (hadamard) gating."""

    def __init__(
        self,
        dim_f: int,
        dim_t: int,
        act: nn.Module = nn.Identity(),
        init_eps: float = 1e-3,
    ):
        """Init method.

        Args:
            dim_f (int): Dimension along frequency axis.
            dim_t (int): Dimension along temporal axis.
            act (nn.Module, optional): Activation function. Defaults to nn.Identity().
            init_eps (float, optional): Weight init. Defaults to 1e-3.
        """
        super().__init__()

        self.act = act
        dim_out = dim_f // 2
        self.norm = nn.LayerNorm(dim_out)
        self.proj = nn.Conv1d(dim_t, dim_t, 1)

        init_eps /= dim_t
        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(self, x):
        x_r, x_g = x.chunk(2, dim=-1)
        return x_r * self.act(self.proj(self.norm(x_g)))


class gMLPBlock(nn.Module):
    """Gated-MLP block with frequency and temporal projections."""

    def __init__(
        self, dim_f: int, dim_f_proj: int, dim_t: int, act: nn.Module = nn.Identity()
    ):
        """Gated-MLP with frequency and temporal projections.

        Args:
            dim_f (int): Size of frequency embeddings.
            dim_f_proj (int): Projection dim across frequency axis.
            dim_t (int): Size along temporal axis.
            act (nn.Module, optional): Activation function. Defaults to nn.Identity().
        """
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(dim_f, dim_f_proj), nn.GELU())
        self.tgu = TemporalGatingUnit(dim_f_proj, dim_t, act)
        self.proj_out = nn.Linear(dim_f_proj // 2, dim_f)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.tgu(x)
        x = self.proj_out(x)
        return x
