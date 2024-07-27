import os
import sys
import torch
import argparse
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("/src/")

from utils import config


class PointWiseFeedForward(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 2048,
        dropout: float = 0.1,
        display: bool = False,
    ):
        super(PointWiseFeedForward, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.display = display

        self.layers = list()

        for index in range(2):
            self.layers.append(
                nn.Linear(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    bias=False,
                )
            )

            self.in_features = self.out_features
            self.out_features = in_features

            if index == 0:
                self.layers.append(nn.ReLU(inplace=True))
                self.layers.append(nn.Dropout1d(p=self.dropout))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.model(x)

        else:
            raise TypeError("Input type is not a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feed Forward Network for Transformer".title()
    )
    parser.add_argument(
        "--in_features", type=int, default=512, help="Input features".capitalize()
    )
    parser.add_argument(
        "--out_features", type=int, default=2048, help="Output features".capitalize()
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate".capitalize(),
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=False,
        help="Display the model".capitalize(),
    )

    args = parser.parse_args()

    net = PointWiseFeedForward(
        in_features=args.in_features, out_features=args.out_features
    )

    assert net(torch.randn((40, 200, args.in_features))).size() == (
        40,
        200,
        args.in_features,
    )
    if args.display:
        print(summary(model=net, input_size=(200, 512)))

        path = config()["path"]["FILES_PATH"]

        draw_graph(
            model=net, input_data=torch.randn((40, 200, 512))
        ).visual_graph.render(
            filename=os.path.join(path, "feedforward_network"), format="png"
        )
