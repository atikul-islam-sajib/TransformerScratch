import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("/src/")

from positional_encoding import PositionalEncoding


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int = 1000,
        sequence_length: int = 200,
        dimension: int = 100,
    ):
        super(EmbeddingLayer, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.model_dimension = dimension

        self.embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size, embedding_dim=self.model_dimension
        )

        self.positional_encoding = PositionalEncoding(
            sequence_length=self.sequence_length, dimension=self.model_dimension
        )

    def forward(self, tokenize: torch.Tensor):
        if isinstance(tokenize, torch.Tensor):
            x = self.embedding(tokenize)
            return x + self.positional_encoding(x)

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    embedding = EmbeddingLayer(vocabulary_size=100, sequence_length=200, dimension=512)
    input_ids = torch.randint(0, 100, (400, 200))

    print(embedding(input_ids).size())
