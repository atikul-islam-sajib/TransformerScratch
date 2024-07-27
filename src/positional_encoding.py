import sys
import math
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn

sys.path.append("/src/")


class PositionalEncoding(nn.Module):
    def __init__(
        self, sequence_length: int = 200, dimension: int = 512, constant: int = 10000
    ):
        super(PositionalEncoding, self).__init__()

        self.sequence_length = sequence_length
        self.model_dimension = dimension
        self.constant = constant

        self.position_encode = torch.ones((sequence_length, dimension))

        for position in tqdm(range(self.sequence_length)):
            for index in range(self.model_dimension):
                if index % 2 == 0:
                    self.position_encode[position, index] = math.sin(
                        position / self.constant ** (2 * index / self.model_dimension)
                    )
                elif index % 2 != 0:
                    self.position_encode[position, index] = math.cos(
                        position / self.constant ** (2 * index / self.model_dimension)
                    )

        self.register_buffer("position_encoding", self.position_encode.unsqueeze(0))

        print("Positional Encoding initialized".capitalize())

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.position_encode[:, : x.shape[-1]]

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Positional Encoder for Transformer".title()
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=200,
        help="Define the sequence length".capitalize(),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="Define the dimension of the model".capitalize(),
    )

    args = parser.parse_args()

    sequence_length = args.seq_length
    model_dimension = args.dimension

    positional_encode = PositionalEncoding(
        sequence_length=sequence_length, dimension=model_dimension
    )

    assert positional_encode(
        torch.randn((sequence_length, model_dimension))
    ).size() == (
        sequence_length,
        model_dimension,
    )
