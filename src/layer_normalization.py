import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("/src/")


class LayerNormalization(nn.Module):
    def __init__(self, normalized_shape: int = 512, epsilon: float = 1e-5):
        super(LayerNormalization, self).__init__()

        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

        self.gamma = nn.Parameter(
            data=torch.ones((normalized_shape,)), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.zeros((normalized_shape,)), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            self.mean = torch.mean(x, dim=-1)
            self.variance = torch.var(x, dim=-1)

            self.mean = self.mean.unsqueeze(-1)
            self.variance = self.variance.unsqueeze(-1)

            return (x - self.mean) / torch.sqrt(
                self.variance + self.epsilon
            ) * self.gamma + self.beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer Normalization for transformer".title()
    )
    parser.add_argument(
        "--normalized_shape",
        type=int,
        default=512,
        help="The normalized shape of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="The epsilon value for the variance".capitalize(),
    )

    args = parser.parse_args()

    normalized_shape = args.normalized_shape
    epsilon = args.epsilon

    layer_norm = LayerNormalization(normalized_shape=normalized_shape, epsilon=epsilon)

    assert layer_norm(torch.randn((40, 200, 512))).size() == (
        40,
        200,
        normalized_shape,
    ), "Layer Normalization is not working properly, check the dimensions".title()
