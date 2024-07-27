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

        assert (
            self.dimension % self.heads == 0
        ), "Dimension must be divisible by heads".title()

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
            self.mask = mask

            KV = self.KV(x)
            Q = self.Q(y)

            self.key, self.value = torch.chunk(input=KV, chunks=2, dim=-1)
            self.query = Q

            self.key = self.key.view(
                self.key.size(0),
                self.key.size(1),
                self.heads,
                self.dimension // self.heads,
            )
            self.value = self.value.view(
                self.value.size(0),
                self.value.size(1),
                self.heads,
                self.dimension // self.heads,
            )
            self.query = self.query.view(
                self.query.size(0),
                self.query.size(1),
                self.heads,
                self.dimension // self.heads,
            )

            self.key = self.key.permute(0, 2, 1, 3)
            self.value = self.value.permute(0, 2, 1, 3)
            self.query = self.query.permute(0, 2, 1, 3)

            result = scaled_dot_product(
                query=self.query,
                key=self.key,
                value=self.value,
                mask=self.mask,
                type="target",
            )

        else:
            raise TypeError("x and y must be torch.Tensor".capitalize())


if __name__ == "__main__":
    attention = MultiCrossAttentionLayer(
        dimension=512,
        heads=8,
        dropout=0.1,
    )

    print(attention(torch.randn((40, 200, 512)), torch.randn((40, 200, 512))).size())
