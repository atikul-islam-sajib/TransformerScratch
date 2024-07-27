import os
import sys
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from torchview import draw_graph

sys.path.append("/src/")

from utils import config
from encoder_block import EncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 8,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        display: bool = False,
    ):

        super(TransformerEncoder, self).__init__()

        self.dimension = d_model
        self.heads = nhead
        self.feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.epsilon = layer_norm_eps
        self.display = display

        self.encoder = nn.Sequential(
            *[
                EncoderBlock(
                    dimension=self.dimension,
                    heads=self.heads,
                    feedforward=self.feedforward,
                    dropout=self.dropout,
                    epsilon=self.epsilon,
                )
                for _ in tqdm(range(self.num_encoder_layers))
            ]
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            for layer in self.encoder:
                x = layer(x, mask)
            return x

        else:
            raise TypeError("Input must be a tensor".capitalize())

    @staticmethod
    def display_parameters(model: nn.Module):
        if isinstance(model, TransformerEncoder):
            return sum(params.numel() for params in model.parameters())

        else:
            raise TypeError("Input must be a transformer encoder".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encoder for Transformer".title())
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension of the model".capitalize()
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of heads".capitalize()
    )
    parser.add_argument(
        "--feedforward",
        type=int,
        default=2048,
        help="Feedforward dimension".capitalize(),
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate".capitalize()
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-6, help="Epsilon for LayerNorm".capitalize()
    )
    parser.add_argument(
        "--display", type=bool, default=False, help="Display the model".capitalize()
    )

    args = parser.parse_args()

    d_model = args.d_model
    nheads = args.nhead
    feedforward = args.feedforward
    dropout = args.dropout
    epsilon = args.epsilon
    display = args.display

    input = torch.randn((40, 200, d_model))
    masked = torch.ones((40, 200))

    encoderTransformer = TransformerEncoder(
        d_model=d_model,
        nhead=nheads,
        dim_feedforward=feedforward,
        dropout=dropout,
        layer_norm_eps=epsilon,
        display=display,
    )

    assert encoderTransformer(input, masked).size() == (
        40,
        200,
        d_model,
    ), "Transformer Encoder block is not working properl as dimension is not equal"

    masked = None

    assert encoderTransformer(input).size() == (
        40,
        200,
        d_model,
    ), "Transformer Encoder block is not working properl as dimension is not equal"

    if display:
        print(
            f"Total parameters of the transformer encoder {TransformerEncoder.display_parameters(model=encoderTransformer)}"
        )

        try:
            path = config()["path"]["FILES_PATH"]

            draw_graph(
                model=encoderTransformer, input_data=torch.randn((40, 200, d_model))
            ).visual_graph.render(
                filename=os.path.join(path, "encoderTransformer"), format="png"
            )

        except Exception as e:
            print("An error occurred: ", e)

        else:
            print(f"Encoder Transformer graph saved successfully in the path {path}")
