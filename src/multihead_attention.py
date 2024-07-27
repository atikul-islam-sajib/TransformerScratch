import sys
import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dimension=512, heads: int = 8, dropout: float = 0.1, mask=None):
        super(MultiHeadAttentionLayer, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout
        self.mask = mask

        assert (
            self.dimension % self.heads == 0
        ), "Dimension must be divisible by heads".title()

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=3 * self.dimension, bias=False
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            QKV = self.QKV(x)

            self.query, self.key, self.values = torch.chunk(input=QKV, chunks=3, dim=-1)

            self.query = self.query.view(
                self.query.size(0),
                self.query.size(1),
                self.heads,
                self.dimension // self.heads,
            )
            self.key = self.key.view(
                self.key.size(0),
                self.key.size(1),
                self.heads,
                self.dimension // self.heads,
            )
            self.values = self.values.view(
                self.values.size(0),
                self.values.size(1),
                self.heads,
                self.dimension // self.heads,
            )

            self.query = self.query.permute(0, 2, 1, 3)
            self.key = self.key.permute(0, 2, 1, 3)
            self.values = self.values.permute(0, 2, 1, 3)

            return self.values

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    multihead = MultiHeadAttentionLayer(dimension=512, heads=8, dropout=0.1, mask=None)

    print(multihead(torch.randn(40, 200, 512)).size())
