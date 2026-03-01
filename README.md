# Mini-LLM

A minimal implementation of a GPT-style language model (decoder-only Transformer) using PyTorch. This project is built from scratch to demonstrate the fundamental components of modern Large Language Models: word/token embeddings, positional embeddings, multi-head self-attention, and feed-forward neural networks.

## Features

- **Custom Transformer Architecture:** Implements a miniature version of the GPT architecture (`MiniGPT`).
- **Multi-Head Self-Attention:** Includes masked self-attention to prevent the model from looking ahead in the sequence.
- **Custom Tokenizer:** Contains a simple tokenization and vocabulary building script.
- **Text Generation:** Scripts to generate conversational text with options like temperature control and repetition penalty.

## What is an LLM?

A Large Language Model (LLM) is an artificial intelligence model designed to understand and generate human language. At its core, an LLM is essentially a sophisticated "next-word prediction" engine. Given a sequence of text, it predicts what words (or parts of words, called tokens) are most likely to follow.

Modern LLMs are built on the **Transformer** architecture, which allows the model to process sequences of data efficiently and capture long-range dependencies within text using a mechanism called "attention."

## How this codebase implements an LLM

This repository is a condensed, from-scratch implementation of an LLM, specifically structured similar to the GPT (Generative Pre-trained Transformer) family. It contains all the foundational pieces of a real-world LLM, scaled down for educational purposes.

Here is how the concepts map to the code:

1. **Tokens and Vocabulary (`tokenizer.py`)**  
   Language models cannot read raw text; they read numbers. `tokenizer.py` breaks down the training dataset (`data.txt`) into small chunks (words/tokens), builds a vocabulary mapping every unique word to an integer (`stoi` and `itos`), and translates the text into a sequence of numbers.
   
2. **Context Window (`config.py - block_size`)**  
   This dictates how far back the model can "look" to predict the next word. A `block_size` of 128 means the model uses up to the previous 128 tokens to predict the 129th.
   
3. **Embeddings (`model.py - token_embedding & position_embedding`)**  
   When an integer token is fed into the model, it is converted into a continuous, high-dimensional vector. 
   - The *Token Embedding* captures the semantic meaning of the word.
   - The *Position Embedding* tells the model *where* the word is in the sequence (since Transformers process data in parallel, not sequentially).

4. **Multi-Head Self-Attention (`model.py - MultiHeadAttention & Head`)**  
   This is the core of the Transformer. The "Attention" mechanism allows each token to look at other tokens in the sequence and determine which ones are relevant to understanding the current context. A "Masked" approach (`self.tril`) is used here, meaning tokens can only attend to previous tokens, stopping the model from "cheating" by looking into the future during training.

5. **Feed Forward Networks (`model.py - FeedForward`)**  
   After tokens communicate with each other via attention, they independently pass through a Feed-Forward neural network to process the information they've gathered.

6. **Transformer Blocks (`model.py - Block`)**  
   The Attention and Feed-Forward layers are wrapped together into a `Block`, along with Layer Normalization and residual connections to stabilize training. These blocks are stacked sequentially (determined by `n_layer` in `config.py`).

7. **Next Token Prediction (`model.py - lm_head`)**  
   Finally, the output of the transformer blocks passes through a final linear layer (`lm_head`) to produce "logits"—raw numbers representing the likelihood of every word in the vocabulary being the next word. During training (`train.py`), these logits are compared to the actual next word using Cross-Entropy loss.

8. **Generation / Inference (`generate.py`)**  
   During generation, the model takes a prompt, converts it to tokens, and outputs logits. Instead of strictly picking the word with the highest probability, it uses strategies like `temperature` scaling and `torch.multinomial` to introduce controlled randomness and creativity into the generated text.

## Project Structure


- `model.py` - Core PyTorch implementation of the Transformer model (`Head`, `MultiHeadAttention`, `FeedForward`, `Block`, and `MiniGPT`).
- `config.py` - Configuration and hyperparameters for the model (e.g., `batch_size`, `n_embd`, `n_head`, `n_layer`).
- `train.py` - Script for training the model on the provided dataset. It handles batching, loss calculation (Cross Entropy), and optimizing.
- `generate.py` - Script for inference/text generation with advanced decoding techniques like top-k sampling, temperature scaling, and repetition penalties.
- `tokenizer.py` - Custom tokenization utilities.
- `visualize.py` - Script to visualize aspects of the model/data.
- `data.txt` - Corpus used for training the model.

## Requirements

Ensure you have Python installed, then install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch>=2.0`
- `numpy>=1.24`
- `matplotlib>=3.7`
- `scikit-learn>=1.3`

## Usage

### 1. Training the Model

To train the language model from scratch, run the `train.py` script. The model will tokenize the text in `data.txt`, build a vocabulary, and train for the number of iterations specified in `config.py`.

```bash
python train.py
```
Upon completion, the trained weights will be saved to `model.pt`.

### 2. Generating Text

Once trained, you can generate text by running:

```bash
python generate.py
```
This script demonstrates an interaction by taking a predefined prompt (e.g., "Agent1: Explain focus") and generating the continuation, handling decoding and formatting of the output token by token.

## Configuration

You can easily scale the model or adjust training parameters in `config.py`. Default settings are:
- `batch_size`: 64
- `block_size`: 128 (context window size)
- `max_iters`: 10000
- `learning_rate`: 3e-4
- `n_embd`: 64
- `n_head`: 4
- `n_layer`: 2
- `dropout`: 0.2

## Hardware Acceleration
The model checks for Apple Silicon (`mps`) or defaults to `cpu` automatically, allowing for faster training and inference on compatible Mac devices.
