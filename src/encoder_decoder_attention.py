import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("/src/")

from transformer_attention import scaled_dot_product


class MultiCrossAttentionLayer(nn.Module):
    def __init__(self, dimension: int = 512, heads: int = 8, dropout: float = 0.1):
        super(MultiCrossAttentionLayer, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout

        self.KV = nn.Linear(
            in_features=self.dimension, out_features=2 * self.dimension, bias=False
        )
        self.Q = nn.Linear(
            in_features=self.dimension, out_features=self.dimension, bias=False
        )

        self.layer = nn.Linear(
            in_features=self.dimension, out_features=self.dimension, bias=False
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            pass

        else:
            raise TypeError("x and y must be torch.Tensor".capitalize())
