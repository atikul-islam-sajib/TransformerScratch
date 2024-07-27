import os
import sys
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from torchview import draw_graph

sys.path.append("/src/")

from utils import config
from decoder_block import DecoderBlock


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        num_decoder_layers: int = 6,
        display: bool = False,
    ):
        super(TransformerDecoder, self).__init__()

        self.dimension = d_model
        self.heads = nhead
        self.feedforward = dim_feedforward
        self.dropout = dropout
        self.epsilon = layer_norm_eps
        self.number_of_layers = num_decoder_layers
        self.display = display

        self.decoder = nn.Sequential(
            *[
                DecoderBlock(
                    dimension=self.dimension,
                    heads=self.heads,
                    feedforward=self.feedforward,
                    dropout=self.dropout,
                    epsilon=self.epsilon,
                )
                for _ in tqdm(range(self.number_of_layers))
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor) and (isinstance(y, torch.Tensor)):
            for layer in self.decoder:
                y = layer(x=x, y=y, mask=mask)

            return y

        else:
            raise TypeError(
                "Input and output must be of type torch.Tensor".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decoder layer for the transformer".title()
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension of the model".capitalize()
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=6,
        help="Number of decoder layers".capitalize(),
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=2048,
        help="Dimension of the feedforward layer".capitalize(),
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate".capitalize()
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        default=1e-6,
        help="Layer norm epsilon".capitalize(),
    )
    parser.add_argument(
        "--display", type=bool, default=False, help="Display the model".capitalize()
    )

    args = parser.parse_args()
    d_model = args.d_model
    num_decoder_layers = args.num_decoder_layers
    dim_feedforward = args.dim_feedforward
    dropout = args.dropout
    layer_norm_eps = args.layer_norm_eps

    decoderTransformer = TransformerDecoder(
        d_model=d_model,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        layer_norm_eps=layer_norm_eps,
    )

    X = torch.randn((40, 200, d_model))
    y = torch.randn((40, 200, d_model))
    padding_masked = torch.randn((40, 200))

    assert decoderTransformer(X, y, padding_masked).size() == (
        40,
        200,
        d_model,
    ), "Dimension mismatch in the decoder".title()

    if args.display:
        path = config()["path"]["FILES_PATH"]

        draw_graph(
            model=decoderTransformer, input_data=(X, y, padding_masked)
        ).visual_graph.render(
            filename=os.path.join(path, "decoderTransformer"), format="png"
        )

        print(f"Decoder model saved in the path {path}")
