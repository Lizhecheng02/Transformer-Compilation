from transformers import GPT2LMHeadModel, GPT2Tokenizer
from chatdata import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({
    'pad_token': '<pad>',
    'bos_token': '<startofstring>',
    'eos_token': '<endofstring>'
})
tokenizer.add_tokens(['<bot>:'])

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

chatdata = ChatData('Transformer\GPT2\Data\chat_data.json',
                    tokenizer=tokenizer)
chatdata = DataLoader(chatdata, batch_size=64)

optim = Adam(model.parameters(), lr=1e-3)


def train(chatdata, model, optim):
    epochs = 20
    for i in range(epochs):
        losses = []
        for X, y in tqdm(chatdata):
            X = X.to(device)
            y = y.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=y, labels=X).loss
            losses.append(loss)
            loss.backward()
            optim.step()
        print(f'Epoch: {i + 1}  Loss: {sum(losses) / len(losses)}')
        losses = []
        torch.save(model.state_dict(), 'Transformer\GPT2\gpt2_model.pt')
        print(infer('How are you'))


def infer(inp):
    inp = '<startofstring> ' + inp + ' <bot>: '
    inp = tokenizer(inp, return_tensors='pt')
    X = inp['input_ids'].to(device)
    y = inp['attention_mask'].to(device)
    output = model.generate(X, attention_mask=y, max_new_tokens=48)
    output = tokenizer.decode(output[0])
    return output


model.train()
train(chatdata=chatdata, model=model, optim=optim)

print("Let's chat ! (typr 'quit' to exit)")
while True:
    inp = input()
    if inp == 'quit':
        break
    print(infer(inp))
