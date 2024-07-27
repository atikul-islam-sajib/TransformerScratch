import sys
import torch
from transformers import AutoTokenizer

sys.path.append("/src/")

from embedding_layer import EmbeddingLayer
from torch.utils.data import DataLoader, TensorDataset
from encoder import TransformerEncoder

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


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokenizer_inputs = tokenizer(
    english,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    max_length=200,
    return_attention_mask=True,
)

input_ids = tokenizer_inputs["input_ids"]
vocab_size = tokenizer.vocab_size
attention_mask = tokenizer_inputs["attention_mask"]

embedding = EmbeddingLayer(
    vocabulary_size=vocab_size, sequence_length=200, dimension=512
)

embedded = embedding(input_ids)
datasets = TensorDataset(embedded, attention_mask)
dataloader = DataLoader(datasets, batch_size=4)

X, attention = next(iter(dataloader))

encoder = TransformerEncoder(
    d_model=512,
    nhead=8,
    dim_feedforward=20248,
    num_encoder_layers=2,
    dropout=0.1,
)

print(encoder(X, attention).size())
