import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import MiniGPT, device

with open("data.txt","r",encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

itos = {i:ch for i,ch in enumerate(chars)}

model = MiniGPT(vocab_size).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))

embeddings = model.token_embedding.weight.detach().cpu().numpy()

tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10,10))
for i, char in enumerate(chars):
    x, y = emb_2d[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, char)

plt.title("Token Embedding Map")
plt.show()