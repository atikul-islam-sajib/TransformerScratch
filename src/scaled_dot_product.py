import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("/src/")

from mask import padding_mask


def scaled_dot_product(
    query: torch.Tensor, key: torch.Tensor, values: torch.Tensor, mask=None
):
    result = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
        (query.size(1) * query.size(3))
    )
    if mask is not None:
        result += padding_mask(mask=mask)

    result = torch.softmax(input=result, dim=-1)

    result = torch.matmul(result, values)

    return result


if __name__ == "__main__":
    scaled = scaled_dot_product(
        torch.randn(40, 8, 200, 64),
        torch.randn(40, 8, 200, 64),
        torch.randn(40, 8, 200, 64),
    )

    print(scaled.size())
