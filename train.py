import torch
from model import MiniGPT, device
from config import *
from tokenizer import build_vocab, tokenize

# Load data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build vocabulary from words
stoi, itos = build_vocab(text)
vocab_size = len(stoi)

# Encode full dataset
tokens = tokenize(text)
encode = lambda s: [stoi[w] for w in tokenize(s)]
data = torch.tensor(encode(text), dtype=torch.long)

# Batch generator
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Create model
model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 200 == 0:
        print(iter, loss.item())

# Save model
torch.save(model.state_dict(), "model.pt")

print("Training complete!")