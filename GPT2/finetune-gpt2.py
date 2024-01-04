import pandas as pd
import datasets
import time
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

data_sample = load_dataset("QuyenAnhDE/Diseases_Symptoms")
print("Data sample:", data_sample)

updated_data = [
    {
        "Name": item["Name"],
        "Symptoms": item["Symptoms"]
    } for item in data_sample["train"]
]
df = pd.DataFrame(updated_data)
print(df.head())

df["Symptoms"] = df["Symptoms"].apply(lambda x: ", ".join(x.split(", ")))
print(df.head())

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    try:
        device = torch.device("mps")
    except Exception:
        device = torch.device("cpu")

print("Device:", device)

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
print(model)


BATCH_SIZE = 8


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.labels = df.columns
        self.data = df.to_dict(orient="records")
        self.tokenizer = tokenizer

        x = self.fittest_max_length(df)
        self.max_length = x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][self.labels[0]]
        y = self.data[idx][self.labels[1]]
        text = f"{x} | {y}"
        tokens = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )
        return tokens

    def fittest_max_length(self, df):
        max_length = max(len(max(df[self.labels[0]], key=len)), len(
            max(df[self.labels[1]], key=len)))
        x = 2
        while x < max_length:
            x = x * 2
        return x


data_sample = CustomDataset(df, tokenizer)
print(data_sample)

train_size = int(0.8 * len(data_sample))
valid_size = len(data_sample) - train_size
train_data, valid_data = random_split(data_sample, [train_size, valid_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

num_epochs = 10
batch_size = BATCH_SIZE
model_name = "distilgpt2"
gpu = 0

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
tokenizer.pad_token = tokenizer.eos_token

results = pd.DataFrame(
    columns=[
        "epoch", "transformer", "batch_size", "gpu",
        "training_loss", "validation_loss", "epoch_duration_sec"
    ]
)

for epoch in range(num_epochs):
    start_time = time.time()

    model.train()

    epoch_training_loss = 0.0
    train_iterator = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch + 1} / {num_epochs} Batch Size: {batch_size}, Transformer: {model_name}"
    )

    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch["input_ids"].squeeze(1).to(device)
        targets = inputs.clone()
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_iterator.set_postfix({"Training Loss": loss.item()})
        epoch_training_loss += loss.item()

    avg_epoch_training_loss = epoch_training_loss / len(train_loader)

    model.eval()

    epoch_validation_loss = 0.0
    total_loss = 0.0

    valid_iterator = tqdm(
        valid_loader,
        desc=f"Validation Epoch {epoch + 1} / {num_epochs}"
    )

    with torch.no_grad():
        for batch in valid_loader:
            inputs = batch["input_ids"].squeeze(1).to(device)
            targets = inputs.clone()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            total_loss += loss
            valid_iterator.set_postfix({"Validation Loss": loss.item()})
            epoch_validation_loss += loss.item()

    avg_epoch_validation_loss = epoch_validation_loss / len(valid_loader)

    end_time = time.time()
    epoch_duration_time = end_time - start_time

    new_row = {
        "transformer": model_name,
        "batch_size": batch_size,
        "gpu": gpu,
        "epoch": epoch + 1,
        "training_loss": avg_epoch_training_loss,
        "validation_loss": avg_epoch_validation_loss,
        "epoch_duration_sec": epoch_duration_time
    }

    results.loc[len(results)] = new_row
    print(
        f"Epoch: {epoch + 1}, Validation Loss: {total_loss / len(valid_loader)}")

input_str = "Kidney Failure"
input_ids = tokenizer.encode(input_str, return_tensors="pt").to(device)

output = model.generate(
    inputs=input_ids,
    max_length=128,
    num_return_sequences=1,
    do_sample=True,
    top_k=8,
    top_p=0.95,
    temperature=0.2,
    repetition_penalty=1.2
)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)

torch.save(model, "finetune-gpt2-model.pt")
