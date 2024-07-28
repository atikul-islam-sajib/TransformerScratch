import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from transformer_attention import scaled_dot_product


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dimension=512, heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttentionLayer, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout

        assert (
            self.dimension % self.heads == 0
        ), "Dimension must be divisible by heads".title()

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=3 * self.dimension, bias=False
        )
        self.layer = nn.Linear(
            in_features=self.dimension, out_features=self.dimension, bias=False
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            self.mask = mask

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

            try:
                self.attention = scaled_dot_product(
                    query=self.query, key=self.key, values=self.values, mask=self.mask
                )

            except Exception as e:
                print("An error occured : {}".format(e))

            else:
                self.attention = self.attention.view(
                    self.attention.size(0),
                    self.attention.size(2),
                    self.attention.size(1) * self.attention.size(3),
                )

                return self.layer(self.attention)

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiHeadAttention Layer for Transformer".title()
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="Dimension of the input tensor".title(),
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of heads for the multihead attention".title(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for the multihead attention".title(),
    )

    args = parser.parse_args()

    dimension = args.dimension
    heads = args.heads
    dropout = args.dropout

    attention = MultiHeadAttentionLayer(
        dimension=dimension, heads=heads, dropout=dropout
    )

    input = torch.randn((40, 200, dimension))
    masked = torch.ones((40, 200))

    assert attention(input, masked).size() == (
        40,
        200,
        dimension,
    ), "Dimension of the output tensor must be equal to the input dimension".capitalize()

    masked = None

    assert attention(input).size() == (
        40,
        200,
        dimension,
    ), "Dimension of the output tensor must be equal to the input dimension".capitalize()
