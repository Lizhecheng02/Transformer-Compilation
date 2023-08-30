import nltk
import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import spacy
import string
import regex as re

from model import BERTModel
from sklearn import model_selection, metrics
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def run():
    dfx = pd.read_csv(config.TRAINING_FILE).fillna('none')
    dfx['sentiment'] = dfx['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    df_train, df_valid = model_selection.train_test_split(dfx, test_size=0.1, random_state=42,
                                                          stratify=dfx['sentiment'].values)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERT_Dataset(review=df_train['review'].values, target=df_train['sentiment'].values)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=2)
    valid_dataset = dataset.BERT_Dataset(review=df_valid['review'].values, target=df_valid['sentiment'].values)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2)

    device = config.DEVICE
    model = BERTModel()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.001,
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        }
    ]

    num_train_steps = int((len(df_train)) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    best_acc = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dataloader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_dataloader, model, device)
        outputs = np.array(outputs) >= 0.5
        acc = metrics.accuracy_score(targets, outputs)
        print(f'Accuracy Score = {acc}')
        if acc > best_acc:
            torch.save(model.state_dict(), './Model/sentiment_model.pth')
            best_acc = acc


if __name__ == '__main__':
    run()
