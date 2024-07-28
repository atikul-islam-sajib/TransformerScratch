import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.append("/src/")

from utils import english, german, dump, config
from embedding_layer import EmbeddingLayer
from torch.utils.data import DataLoader, TensorDataset
from encoder import TransformerEncoder
from decoder import TransformerDecoder

english = [
    "The sun is shining brightly today",
    "I enjoy reading books on rainy afternoons",
    "The cat sat on the windowsill watching the birds",
    "She baked a delicious chocolate cake for dessert",
    "We went for a long walk in the park yesterday",
]

german = [
    "Die Sonne scheint heute hell",
    "Ich lese gerne Bücher an regnerischen Nachmittagen",
    "Die Katze saß auf der Fensterbank und beobachtete die Vögel",
    "Sie hat einen leckeren Schokoladenkuchen zum Nachtisch gebacken",
    "Wir haben gestern einen langen Spaziergang im Park gemacht",
]


# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# tokenizer_inputs = tokenizer(
#     english,
#     padding="max_length",
#     truncation=True,
#     return_tensors="pt",
#     max_length=200,
#     return_attention_mask=True,
# )

# input_ids = tokenizer_inputs["input_ids"]
# vocab_size = tokenizer.vocab_size
# attention_mask = tokenizer_inputs["attention_mask"]

# datasets = TensorDataset(input_ids, attention_mask)
# dataloader = DataLoader(datasets, batch_size=2, shuffle=True)

# inputs, padding_masked = next(iter(dataloader))


# embedding = EmbeddingLayer(
#     vocabulary_size=vocab_size, sequence_length=200, dimension=512
# )

# print(embedding(inputs).size())

# embedded = embedding(input_ids)
# datasets = TensorDataset(embedded, attention_mask)
# dataloader = DataLoader(datasets, batch_size=4)

# X, attention = next(iter(dataloader))

# encoder = TransformerEncoder(
#     d_model=512,
#     nhead=8,
#     dim_feedforward=20248,
#     num_encoder_layers=2,
#     dropout=0.1,
# )

# print(encoder(X, attention).size())


class Tokenizer:
    def __init__(
        self,
        text: list,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        max_length: int = 200,
        batch_size: int = 4,
        return_attention_mask: bool = True,
    ):
        self.text = text
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.max_length = max_length
        self.batch_size = batch_size
        self.return_attention_mask = return_attention_mask

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            print("An error occurred: ", e)

    def tokenize_text(self):
        if isinstance(self.text, list):
            tokenizer_inputs = self.tokenizer(
                self.text,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors=self.return_tensors,
                max_length=self.max_length,
                return_attention_mask=self.return_attention_mask,
            )

            return {
                "tokenize_inputs": tokenizer_inputs,
                "input_ids": tokenizer_inputs["input_ids"],
                "attention_mask": tokenizer_inputs["attention_mask"],
                "vocab_size": self.tokenizer.vocab_size,
            }

        else:
            raise TypeError("Input must be a list of strings".capitalize())

    def create_dataloader(self):
        try:
            tokenize = self.tokenize_text()

            input_ids = tokenize["input_ids"]
            attention_mask = tokenize["attention_mask"]
            vocab_size = tokenize["vocab_size"]

            datasets = TensorDataset(input_ids, attention_mask)
            dataloader = DataLoader(
                dataset=datasets, batch_size=self.batch_size, shuffle=True
            )

            dump(
                value=dataloader,
                filename=os.path.join(
                    config()["path"]["PROCESSED_PATH"], "dataloader.pkl"
                ),
            )

            return {
                "dataloader": dataloader,
                "vocab_size": vocab_size,
            }

        except Exception as e:
            print("An error occurred: ", e)


if __name__ == "__main__":
    tokenizer = Tokenizer(text=english, max_length=200)
    tokenizer = tokenizer.create_dataloader()
