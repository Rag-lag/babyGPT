# babyGPT Shakespeare Text Generator

A PyTorch implementation of a small GPT (Generative Pre-trained Transformer) model that generates Shakespeare-like text. This model learns the statistical patterns of Shakespeare's writing style but generates semantically nonsensical text that mimics the linguistic patterns of Shakespeare's works.

## Overview

This implementation includes:

- A character-level language model based on the transformer architecture
- Multi-head self-attention mechanism
- Positional embeddings
- Layer normalization and dropout for regularization
- Trained on the Tiny Shakespeare dataset

## Requirements

- Python 3.x
- PyTorch
- CUDA-capable GPU (optional, will use CPU if GPU is not available)

## Setup

1. Install the required dependencies:

```bash
pip install torch
```

2. Download the training data:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Model Architecture

- Embedding dimension: 384
- Number of attention heads: 6
- Number of transformer layers: 6
- Dropout rate: 0.2
- Context window size: 256 characters
- Total parameters: ~10M

## Training

The model is trained with the following hyperparameters:

- Batch size: 64
- Learning rate: 3e-4
- Training iterations: 5000
- Evaluation interval: 500 iterations

The training process includes:

- Training/validation split (90/10)
- AdamW optimizer
- Cross-entropy loss function
- Regular evaluation on both training and validation sets

## Usage

To train the model and generate text:

1. Ensure the `input.txt` file containing Shakespeare's text is in the same directory
2. Run the script:

```bash
python gpt.py
```

The script will:

- Train the model for 5000 iterations
- Print training and validation loss every 500 iterations
- Generate 500 tokens of new text after training
- Output the generated text to console

## Output

The model generates character-by-character text that mimics Shakespeare's writing style in terms of:

- Vocabulary and word usage
- Character names and dialogue patterns
- Structural elements like line breaks and spacing

Note: While the generated text maintains Shakespeare's writing style, it does not produce coherent or meaningful sentences. This is expected given the model's size and character-level nature.

## Modifications

To adjust the model's behavior, you can modify these parameters in the code:

- `block_size`: Maximum context length
- `n_embd`: Embedding dimension
- `n_head`: Number of attention heads
- `n_layer`: Number of transformer layers
- `max_new_tokens`: Length of generated text
- `learning_rate`: Training learning rate
- `max_iters`: Number of training iterations

## Acknowledgments

This implementation is inspired by Andrej Karpathy's "Building GPT from Scratch" series and uses the Tiny Shakespeare dataset from his char-rnn repository.
