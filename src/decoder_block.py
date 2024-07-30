import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config
from layer_normalization import LayerNormalization
from feedforward_network import PointWiseFeedForward
from multihead_attention import MultiHeadAttentionLayer
from encoder_decoder_attention import MultiCrossAttentionLayer


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        heads: int = 8,
        feedforward: int = 2048,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        display: bool = False,
    ):
        super(DecoderBlock, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.feedforward = feedforward
        self.dropout = dropout
        self.epsilon = epsilon
        self.display = display

        self.masked_multihead_attention = MultiHeadAttentionLayer(
            dimension=self.dimension,
            heads=self.heads,
            dropout=self.dropout,
        )

        self.layer_norm = LayerNormalization(
            normalized_shape=self.dimension,
            epsilon=self.epsilon,
        )

        self.encoder_deecoder_attention = MultiCrossAttentionLayer(
            dimension=self.dimension,
            heads=self.heads,
            dropout=self.dropout,
        )

        self.feedforward_network = PointWiseFeedForward(
            in_features=self.dimension,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            residual = y

            y = self.masked_multihead_attention(x=y, mask=mask)
            y = torch.dropout(input=y, p=self.dropout, train=self.training)
            y = torch.add(y, residual)
            y = self.layer_norm(y)

            residual = y

            y = self.encoder_deecoder_attention(x=x, y=y, mask=None)
            y = torch.dropout(input=y, p=self.dropout, train=self.training)
            y = torch.add(y, residual)
            y = self.layer_norm(y)

            residual = y

            y = self.feedforward_network(y)
            y = torch.dropout(input=y, p=self.dropout, train=self.training)
            y = torch.add(y, residual)
            y = self.layer_norm(y)

            return y
        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decoder block for the Transformer model".title()
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
        help="Number of attention heads".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate".capitalize(),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Epsilon for the layer normalization".capitalize(),
    )
    parser.add_argument(
        "--feedforward",
        type=int,
        default=2048,
        help="Feedforward dimension".capitalize(),
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=False,
        help="Display the model architecture".capitalize(),
    )

    args = parser.parse_args()

    dimension = args.dimension
    heads = args.heads
    dropout = args.dropout
    epsilon = args.epsilon
    feedforward = args.feedforward

    decoder = DecoderBlock(
        dimension=dimension,
        heads=heads,
        dropout=dropout,
        epsilon=epsilon,
        feedforward=feedforward,
    )
    X = torch.randn((40, 200, dimension))
    y = torch.randn((40, 200, dimension))

    assert decoder(X, y).size() == (
        40,
        200,
        dimension,
    ), "Model output size is incorrect from decoder block".capitalize()

    if args.display:
        path = config()["path"]["FILES_PATH"]

        draw_graph(model=decoder, input_data=(X, y)).visual_graph.render(
            filename=os.path.join(path, "one_decoder"), format="png"
        )

        print(f"Model architecture is saved in the path {path}".capitalize())
