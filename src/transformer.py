import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("/src/")

from encoder import EncoderBlock
from src.decoder_block import DecoderBlock


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
        self.dimension = d_model
        self.heads = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.feedfowrard = dim_feedforward
        self.dropout = dropout
        self.epsilon = layer_norm_eps
