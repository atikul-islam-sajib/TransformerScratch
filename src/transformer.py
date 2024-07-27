import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("/src/")

from utils import config
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-05,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

        self.transformerEncoder = TransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.num_encoder_layers,
            dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.transformerDecoder = TransformerDecoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        encoder_padding_mask=None,
        decoder_padding_mask=None,
    ):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            x = self.transformerEncoder(x=x, mask=encoder_padding_mask)
            x = self.transformerDecoder(x=x, y=y, mask=decoder_padding_mask)

            return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer model".capitalize())
    parser.add_argument(
        "--d_model", type=int, default=512, help="Embedding dimension".capitalize()
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of heads".capitalize
    )
    parser.add_argument(
        "--ffn", type=int, default=2048, help="Feed forward dimension".capitalize()
    )
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=8,
        help="Number of encoder layers".capitalize(),
    )
    parser.add_argument(
        "--decoder_layers",
        type=int,
        default=8,
        help="Number of decoder layers".capitalize(),
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout".capitalize()
    )
    parser.add_argument(
        "--eps", type=float, default=1e-6, help="Layer norm epsilon".capitalize()
    )
    parser.add_argument(
        "--display", type=bool, default=False, help="Display model".capitalize()
    )

    args = parser.parse_args()

    d_model = args.d_model
    nhead = args.nhead
    dim_feedforward = args.ffn
    num_encoder_layers = args.encoder_layers
    num_decoder_layers = args.decoder_layers
    dropout = args.dropout
    layer_norm_eps = args.eps

    transformer = Transformer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_decoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        layer_norm_eps=layer_norm_eps,
    )

    X = torch.randn((40, 200, d_model))
    y = torch.randn((40, 200, d_model))

    encoder_padding_mask = torch.ones((40, 200))
    decoder_padding_mask = torch.ones((40, 200))

    assert (
        transformer(
            x=X,
            y=y,
            encoder_padding_mask=encoder_padding_mask,
            decoder_padding_mask=decoder_padding_mask,
        ).size()
    ) == (
        40,
        200,
        d_model,
    ), "Output of the transformer is not correct as the dimensions are not matching".title()

    if args.display:
        print(
            "Total parameters of the transformer: ",
            sum(params.numel() for params in transformer.parameters()),
        )

        path = config()["path"]["FILES_PATH"]

        draw_graph(
            model=transformer,
            input_data=(X, y, encoder_padding_mask, decoder_padding_mask),
        ).visual_graph.render(filename=os.path.join(path, "Transformer"), format="png")

        print(f"Decoder model saved in the path {path}")
