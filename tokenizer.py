import re
import pickle

def build_vocab(text):
    words = re.findall(r"\w+|[^\w\s]", text)
    vocab = sorted(set(words))

    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for i,w in enumerate(vocab)}

    with open("vocab.pkl","wb") as f:
        pickle.dump((stoi, itos), f)

    return stoi, itos

def load_vocab():
    with open("vocab.pkl","rb") as f:
        return pickle.load(f)

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)