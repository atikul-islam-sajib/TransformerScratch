import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


def padding_mask(mask: torch.Tensor):
    mask = torch.where(mask == 0.0, float("-inf"), mask)
    return mask.unsqueeze(1).unsqueeze(2)


def target_mask(sequence_length: int = 200):
    mask = torch.triu(input=torch.ones((sequence_length, sequence_length)), diagonal=1)
    mask = torch.where(mask == 1.0, float("-inf"), mask)
    return mask


if __name__ == "__main__":
    mask = padding_mask(mask=torch.ones((40, 200)))
    assert mask.size() == (40, 1, 1, 200)

    mask = target_mask(sequence_length=200)
    print(mask.size())
