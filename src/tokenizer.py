import os
import sys
from transformers import AutoTokenizer

sys.path.append("/src/")

from utils import english, german, config, dump
from torch.utils.data import DataLoader, TensorDataset


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
                "tokenizer": self.tokenizer,
                "dataloader": dataloader,
                "vocab_size": vocab_size,
            }

        except Exception as e:
            print("An error occurred: ", e)


if __name__ == "__main__":
    tokenizer = Tokenizer(
        text=english,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=200,
        batch_size=4,
        return_attention_mask=True,
    )

    tokenizer = tokenizer.create_dataloader()

    dataloader = tokenizer["dataloader"]
    vocab_size = tokenizer["vocab_size"]

    assert vocab_size == 30522
