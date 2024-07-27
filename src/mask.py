import sys
import torch
import torch.nn as nn


def padding_mask(mask: torch.Tensor):
    mask = torch.where(mask == 0.0, float("-inf"), mask)
    return mask.unsqueeze(1).unsqueeze(2)


if __name__ == "__main__":
    mask = padding_mask(mask=torch.ones((40, 200)))
    print(mask.size())
