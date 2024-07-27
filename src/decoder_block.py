import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("/src/")

from layer_normalization import LayerNormalization
from multihead_attention import MultiHeadAttentionLayer
from encoder_decoder_attention import MultiCrossAttentionLayer


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

        self.encoder_deecoder_attention = MultiCrossAttentionLayer(
            dimension=self.dimension, heads=self.heads, dropout=self.dropout
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            residual = y

            y = self.masked_multihead_attention(x=y, mask=mask)
            y = torch.dropout(input=y, p=self.dropout, train=self.training)
            y = torch.add(y, residual)
            y = self.layer_norm(y)

            residual = y

            y = self.encoder_deecoder_attention(x=x, y=y)
            y = torch.dropout(input=y, p=self.dropout, train=self.training)
            y = torch.add(y, residual)
            y = self.layer_norm(y)

            return y

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    decoder = DecoderBlock(dimension=512, heads=8, dropout=0.1, epsilon=1e-6)
    print(decoder(torch.randn((40, 200, 512)), torch.randn((40, 200, 512))).size())
