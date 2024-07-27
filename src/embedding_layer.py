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
    parser = argparse.ArgumentParser(
        description="Embedding Layer for Transformer".title()
    )
    parser.add_argument(
        "--vocab_size", type=int, default=100, help="Vocabulary Size".capitalize()
    )
    parser.add_argument(
        "--seq_len", type=int, default=200, help="Sequence Length".capitalize()
    )
    parser.add_argument(
        "--dim", type=int, default=512, help="Dimension of the Model".capitalize()
    )

    args = parser.parse_args()

    sequence_length = args.seq_len
    vocabulary_size = args.vocab_size
    model_dimension = args.dim

    embedding = EmbeddingLayer(
        vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
        dimension=model_dimension,
    )
    input_ids = torch.randint(0, vocabulary_size, (400, sequence_length))

    assert embedding(input_ids).size() == (
        400,
        sequence_length,
        model_dimension,
    ), "Dimension Mismatch in the embedding layer".title()
