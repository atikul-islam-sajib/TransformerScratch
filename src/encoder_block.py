import os
import sys
import torch
import argparse
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("/src/")

from utils import config
from layer_normalization import LayerNormalization
from feedforward_network import PointWiseFeedForward
from multihead_attention import MultiHeadAttentionLayer


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        heads: int = 8,
        feedforward: int = 2048,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        display: bool = False,
    ):
        super(EncoderBlock, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.feedforward = feedforward
        self.dropout = dropout
        self.epsilon = epsilon
        self.display = display

        self.multihead_attention = MultiHeadAttentionLayer(
            dimension=self.dimension, heads=self.heads, dropout=self.dropout
        )

        self.layer_norm = LayerNormalization(
            normalized_shape=self.dimension, epsilon=self.epsilon
        )

        self.feedforward_network = PointWiseFeedForward(
            in_features=self.dimension,
            out_features=self.feedforward,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            self.mask = mask

            residual = x

            x = self.multihead_attention(x=x, mask=self.mask)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)
            x = self.layer_norm(x)

            residual = x

            x = self.feedforward_network(x=x)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(residual, x)
            x = self.layer_norm(x)

            return x

        else:
            raise TypeError("Input must be a tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder Block for Transfomers".title()
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="Dimension of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of heads in the multi-head attention".capitalize(),
    )
    parser.add_argument(
        "--feedfoward",
        type=int,
        default=2048,
        help="Dimension of the feedforward network".capitalize(),
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate".capitalize()
    )
    parser.add_argument(
        "-eps",
        type=float,
        default=1e-6,
        help="Epsilon value for layer normalization".capitalize(),
    )
    parser.add_argument(
        "--display", type=bool, default=False, help="Display the model".capitalize()
    )

    args = parser.parse_args()

    dimension = args.dimension
    heads = args.heads
    dropout = args.dropout
    feedforward = args.feedfoward
    eps = args.eps
    display = args.display

    encoder = EncoderBlock(
        dimension=dimension,
        heads=heads,
        dropout=dropout,
    )
    masked = torch.ones((40, 200))

    assert encoder(torch.randn((40, 200, dimension)), masked).size() == (
        40,
        200,
        dimension,
    ), "Encoder block is not working properl as dimension is not equal".title()

    masked = None

    assert encoder(torch.randn((40, 200, dimension))).size() == (
        40,
        200,
        dimension,
    ), "Encoder block is not working properl as dimension is not equal".title()

    if display:

        path = config()["path"]["FILES_PATH"]

        draw_graph(
            model=encoder, input_data=torch.randn((40, 200, dimension))
        ).visual_graph.render(filename=os.path.join(path, "one_encoder"), format="png")
