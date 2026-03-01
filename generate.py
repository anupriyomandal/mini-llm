import torch
from model import MiniGPT, device
from config import *
from tokenizer import load_vocab, tokenize

# Load vocab
stoi, itos = load_vocab()
vocab_size = len(stoi)

# Decode words
def decode(tokens):
    words = [itos[i] for i in tokens]
    text = ' '.join(words)

    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace(" :", ":")

    return text

# Load model
model = MiniGPT(vocab_size).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))

# Prompt
prompt = "Agent1: Explain focus"

# Tokenize prompt into words
tokens = tokenize(prompt.lower())
encoded = torch.tensor([[stoi.get(w, stoi['.']) for w in tokens]], dtype=torch.long).to(device)

idx = encoded

# Generation loop
for _ in range(50):

    logits, _ = model(idx)

    # Repetition penalty
    repetition_penalty = 1.2
    for token in set(idx[0].tolist()):
        logits[:, :, token] /= repetition_penalty

    logits = logits[:, -1, :]

    temperature = 0.7
    logits = logits / temperature

    k = 5
    topk_vals, topk_idx = torch.topk(logits, k)
    probs = torch.softmax(topk_vals, dim=-1)

    next_token = torch.multinomial(probs, 1)
    idx_next = topk_idx.gather(-1, next_token)

    idx = torch.cat((idx, idx_next), dim=1)
    
    decoded = decode(idx[0].tolist())

    # Stop once next Agent1 begins
    if "Agent1" in decoded.split("Agent2:")[-1]:
        break

full_output = decode(idx[0].tolist())

if "Agent2:" in full_output:
    answer = full_output.split("Agent2:")[-1]

    if "Agent1" in answer:
        answer = answer.split("Agent1")[0]

    print("Agent2:", answer.strip())
else:
    print(full_output)