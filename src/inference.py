"""
This script tests the implementation of a Transformer model from scratch using a dummy dataset. 
It includes tokenization, embedding, and the Transformer model itself. The script ensures that the 
code runs correctly and the Transformer model produces the expected output.

Modules used:
    - os
    - sys
    - transformers.AutoTokenizer
    - utils: Provides the dummy English and German sentences.
    - tokenizer: Custom Tokenizer class for tokenizing and processing text.
    - embedding_layer: Custom EmbeddingLayer class for creating embeddings.
    - transformer: Custom Transformer class for the Transformer model.

The script performs the following steps:
    1. Imports necessary modules and sets up paths.
    2. Checks if the lengths of the English and German sentences are equal.
    3. Initializes Tokenizers and DataLoaders for English and German sentences.
    4. Initializes the Embedding Layer with the appropriate vocabulary size and dimensions.
    5. Initializes the Transformer model with specified hyperparameters.
    6. Tests the Transformer model by feeding it embeddings from the dummy dataset and prints the output size.

Usage:
    This script is intended to verify that the implemented Transformer model works correctly with a dummy dataset.
    You can also use your own embeddings instead of the provided ones.
"""

# Variable values for easy configuration
MAX_LENGTH = 200
BATCH_SIZE = 40
EMBEDDING_DIMENSION = 512
NUM_ENCODER_LAYERS = 8
NUM_DECODER_LAYERS = 8
NUM_HEADS = 8
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
LAYER_NORM_EPS = 1e-5

# Check if all necessary modules are available
try:
    import os
    import sys
    from transformers import AutoTokenizer

    sys.path.append("/src/")

    from utils import english, german
    from tokenizer import Tokenizer
    from embedding_layer import EmbeddingLayer
    from transformer import Transformer
except ImportError as e:
    print(f"Import Error: {e}")

# Ensure that the lengths of sentences match
if len(english) != len(german):
    raise ValueError("Length of the sentences are not equal")

# Initialize Tokenizers and DataLoaders
english_tokenizer = Tokenizer(
    text=english,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
)
english_tokenizer_results = english_tokenizer.create_dataloader()
english_dataloader = english_tokenizer_results["dataloader"]
english_vocab_size = english_tokenizer_results["vocab_size"]

german_tokenizer = Tokenizer(
    text=german,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
)
german_tokenizer_results = german_tokenizer.create_dataloader()
german_dataloader = german_tokenizer_results["dataloader"]
german_vocab_size = german_tokenizer_results["vocab_size"]

# Initialize Embedding Layer
embedding_layer = EmbeddingLayer(
    vocabulary_size=english_vocab_size,
    dimension=EMBEDDING_DIMENSION,
    sequence_length=MAX_LENGTH,
)

# Initialize Transformer
transformer_model = Transformer(
    d_model=EMBEDDING_DIMENSION,
    nhead=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    layer_norm_eps=LAYER_NORM_EPS,
)

# Test the Transformer with embeddings
for (english_batch, english_padding_mask), (german_batch, german_padding_mask) in zip(
    english_dataloader, german_dataloader
):
    english_embeddings = embedding_layer(english_batch)
    german_embeddings = embedding_layer(german_batch)

    transformer_output = transformer_model(
        x=english_embeddings,
        y=german_embeddings,
        encoder_padding_mask=english_padding_mask,
        decoder_padding_mask=german_padding_mask,
    )
    print(transformer_output.size())
    break  # Test with only the first batch


####################################################################################################################
####################################################################################################################
#                            THIS IS ANOTHER APPROACH THAT YOU CAN USE TO RUN THE TRANSFORMER                      #
####################################################################################################################
####################################################################################################################

from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


############################
#          English         #
############################

english_tokenizer = tokenizer(
    english,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    max_length=MAX_LENGTH,
)

print("Tokenized Input IDs:", english_tokenizer["input_ids"].size())
print("Attention Mask:", english_tokenizer["attention_mask"].size())

print("*" * 50, "\n")

english_vocab_size = tokenizer.vocab_size

english_tokenizer_results = TensorDataset(
    english_tokenizer["input_ids"], english_tokenizer["attention_mask"]
)
english_tokenizer_dataloader = DataLoader(
    english_tokenizer_results, batch_size=BATCH_SIZE, shuffle=True
)

############################
#          German          #
############################

german_tokenizer = tokenizer(
    german,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    max_length=MAX_LENGTH,
)

print("Tokenized Input IDs:", german_tokenizer["input_ids"].size())
print("Attention Mask:", german_tokenizer["attention_mask"].size())

print("*" * 50, "\n")

german_vocab_size = tokenizer.vocab_size

german_tokenizer_results = TensorDataset(
    german_tokenizer["input_ids"], german_tokenizer["attention_mask"]
)
german_tokenizer_dataloader = DataLoader(
    german_tokenizer_results, batch_size=BATCH_SIZE, shuffle=True
)

###########################
#         Embedding       #
###########################

assert german_vocab_size == english_vocab_size, "Vocabulary sizes must be equal"

embedding = EmbeddingLayer(
    vocabulary_size=english_vocab_size,
    sequence_length=MAX_LENGTH,
    dimension=EMBEDDING_DIMENSION,
)

# Test the Transformer with embeddings
for (english_batch, english_padding_mask), (german_batch, german_padding_mask) in zip(
    english_tokenizer_dataloader, german_tokenizer_dataloader
):
    english_embeddings = embedding(english_batch)
    german_embeddings = embedding(german_batch)

    transformer_output = transformer_model(
        x=english_embeddings,
        y=german_embeddings,
        encoder_padding_mask=english_padding_mask,
        decoder_padding_mask=german_padding_mask,
    )
    print(transformer_output.size())
    break  # Test with only the first batch
