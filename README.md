# Mini-LLM

A minimal implementation of a GPT-style language model (decoder-only Transformer) using PyTorch. This project is built from scratch to demonstrate the fundamental components of modern Large Language Models: word/token embeddings, positional embeddings, multi-head self-attention, and feed-forward neural networks.

## Features

- **Custom Transformer Architecture:** Implements a miniature version of the GPT architecture (`MiniGPT`).
- **Multi-Head Self-Attention:** Includes masked self-attention to prevent the model from looking ahead in the sequence.
- **Custom Tokenizer:** Contains a simple tokenization and vocabulary building script.
- **Text Generation:** Scripts to generate conversational text with options like temperature control and repetition penalty.

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
