import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("/src/")

from layer_normalization import LayerNormalization
from multihead_attention import MultiHeadAttentionLayer


class EncoderBlock(nn.Module):
    def __init__(self, dimension: int = 512, heads: int = 8, dropout: float = 0.1):
        super(EncoderBlock, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout

        self.multihead_attention = MultiHeadAttentionLayer(
            dimension=self.dimension, heads=self.heads, dropout=self.dropout
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            self.mask = mask

            residual = x

            x = self.multihead_attention(x=x, mask=self.mask)

            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)

            residual = x

        else:
            raise TypeError("Input must be a tensor".capitalize())


if __name__ == "__main__":
    encoder = EncoderBlock(
        dimension=512,
        heads=8,
        dropout=0.1,
    )
    masked = torch.ones((40, 200))
    print(encoder(torch.randn((40, 200, 512)), masked).size())
