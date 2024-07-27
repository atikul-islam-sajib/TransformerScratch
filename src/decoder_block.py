import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("/src/")

from layer_normalization import LayerNormalization
from multihead_attention import MultiHeadAttentionLayer


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        heads: int = 8,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
    ):
        super(DecoderBlock, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout
        self.epsilon = epsilon

        self.masked_multihead_attention = MultiHeadAttentionLayer(
            dimension=self.dimension, heads=self.heads, dropout=self.dropout
        )

        self.layer_norm = LayerNormalization(
            normalized_shape=self.dimension, epsilon=self.epsilon
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            residual = x

            x = self.masked_multihead_attention(x=x, mask=mask)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)
            x = self.layer_norm(x)

            residual = x

            return x

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    decoder = DecoderBlock(dimension=512, heads=8, dropout=0.1, epsilon=1e-6)
    print(decoder(torch.randn((40, 200, 512))).size())
