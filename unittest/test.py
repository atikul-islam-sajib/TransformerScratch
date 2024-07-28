import sys
import math
import torch
import torch.nn as nn
import unittest
from torch.utils.data import DataLoader

sys.path.append("./src/")

from embedding_layer import EmbeddingLayer
from positional_encoding import PositionalEncoding
from feedforward_network import PointWiseFeedForward
from layer_normalization import LayerNormalization
from multihead_attention import MultiHeadAttentionLayer
from encoder import TransformerEncoder


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.vocabulary_size = 1000
        self.sequence_length = 200
        self.dimension = 512
        self.feedforward = 2048
        self.batch_size = 64
        self.nheads = 8

        self.embedding = EmbeddingLayer(
            vocabulary_size=self.vocabulary_size,
            sequence_length=self.sequence_length,
            dimension=self.dimension,
        )

        self.positional_encoding = PositionalEncoding(
            sequence_length=self.sequence_length, dimension=self.dimension
        )

        self.feedforward_network = PointWiseFeedForward(
            in_features=self.dimension, out_features=self.feedforward
        )

        self.layer_norm = LayerNormalization(normalized_shape=self.dimension)

        self.multihead_attention = MultiHeadAttentionLayer(
            dimension=self.dimension,
            heads=self.nheads,
        )

        self.encoder = TransformerEncoder(
            d_model=self.dimension,
            nhead=self.nheads,
            num_encoder_layers=6,
        )

    def tearDown(self):
        pass

    def test_embedding_layer(self):
        inputs = torch.randint(
            0 // self.vocabulary_size,
            self.vocabulary_size,
            (self.vocabulary_size // 2, self.sequence_length),
        )

        self.assertEqual(
            self.embedding(inputs).size(),
            torch.Size(
                [self.vocabulary_size // 2, self.sequence_length, self.dimension]
            ),
        )

    def test_embedding_with_batch_size(self):
        inputs = torch.randint(
            0 // self.vocabulary_size,
            self.vocabulary_size,
            (self.vocabulary_size // 2, self.sequence_length),
        )

        dataloader = DataLoader(inputs, batch_size=self.batch_size, shuffle=False)

        dataset = next(iter(dataloader))

        self.assertEqual(
            self.embedding(dataset).size(),
            torch.Size([self.batch_size, self.sequence_length, self.dimension]),
        )

    def test_positional_encoding(self):

        self.assertTrue(self.dimension % self.nheads == 0)

        query = torch.randn(
            (
                self.batch_size,
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            )
        )
        key = torch.randn(
            (
                self.batch_size,
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            )
        )

        results = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.dimension // self.nheads
        )
        self.assertEqual(
            results.size(),
            torch.Size(
                [
                    self.batch_size,
                    self.nheads,
                    self.sequence_length,
                    self.sequence_length,
                ]
            ),
        )

        positional_encoding = self.positional_encoding(
            x=torch.randn((self.batch_size, self.sequence_length, self.dimension))
        )

        self.assertEqual(
            positional_encoding.size(),
            torch.Size([self.sequence_length, self.dimension]),
        )

    def test_feedforward_network(self):
        inputs = torch.randn((self.batch_size, self.sequence_length, self.dimension))
        result = self.feedforward_network(inputs)

        self.assertEqual(result.size(), inputs.size())

    def test_layer_normalization(self):
        inputs = torch.randn((self.batch_size, self.sequence_length, self.dimension))

        self.assertEqual(self.layer_norm(inputs).size(), inputs.size())

    def test_multihead_attention(self):
        inputs = torch.randn(self.batch_size, self.sequence_length, self.dimension)

        self.assertEqual(self.multihead_attention(inputs).size(), inputs.size())

        encoder_masked = torch.randn(self.batch_size, self.sequence_length)

        self.assertEqual(
            self.multihead_attention(inputs, encoder_masked).size(), inputs.size()
        )

    def test_encoder(self):
        inputs = torch.randn(self.batch_size, self.sequence_length, self.dimension)
        encoder_padding_masked = None

        self.assertEqual(
            self.encoder(x=inputs, mask=encoder_padding_masked).size(), inputs.size()
        )

        encoder_padding_masked = torch.randn(self.batch_size, self.sequence_length)

        self.assertEqual(
            self.encoder(x=inputs, mask=encoder_padding_masked).size(), inputs.size()
        )


if __name__ == "__main__":
    unittest.main()
