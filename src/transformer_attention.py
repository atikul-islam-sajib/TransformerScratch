import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("/src/")

from mask import padding_mask


def scaled_dot_product(
    query: torch.Tensor,
    key: torch.Tensor,
    values: torch.Tensor,
    mask=None,
    type: str = "src",
):
    if (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(values, torch.Tensor)
    ) and (query.size() == key.size() == values.size()):

        result = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            (query.size(1) * query.size(3))
        )

        if (mask is not None) and (type == "src"):
            result += padding_mask(mask=mask)

        elif (mask is not None) and (type == "target"):
            result += padding_mask(mask=mask, type="tgt_mask")

        result = torch.softmax(input=result, dim=-1)

        result = torch.matmul(result, values)

        return result
    else:
        raise TypeError(
            "The query, key, and values must be of type torch.Tensor and same shape".capitalize()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaled dot product for Transformer".title()
    )
    parser.parse_args()

    query = torch.randn((40, 8, 200, 64))
    key = torch.randn((40, 8, 200, 64))
    values = torch.randn((40, 8, 200, 64))

    attention_output = scaled_dot_product(query=query, key=key, values=values)

    assert attention_output.size() == (40, 8, 200, 64)
