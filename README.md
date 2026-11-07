# Character-Level Transformer

This project contains two character-level language model implementations:

- [`bigram.py`](./bigram.py) is a small embedding-based baseline.
- [`v2.py`](./v2.py) is a decoder-only Transformer with causal multi-head self-attention.

The Transformer predicts the next character in a text sequence. It trains from
`input.txt`, which is intentionally excluded from version control; provide your
own text file at that path before running the scripts.

## Requirements and running

Python 3 and PyTorch are required. Install PyTorch using the instructions for
your operating system and hardware from the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/).

From the project directory, run:

```bash
python3 v2.py
```

The script selects CUDA when available and otherwise uses the CPU. It prints
training and validation loss periodically, then generates 500 characters.
`bigram.py` can be run in the same way for the baseline model.

## Transformer architecture (`v2.py`)

### Text representation and data batches

The model builds a vocabulary from the unique characters in `input.txt`.
Characters are assigned integer IDs using a character-to-ID mapping, and
generated IDs are converted back to text with the inverse mapping.

The text is split contiguously: the first 90% is used for training and the
remaining 10% for validation. Each training batch contains random windows of
`BLOCK_SIZE` characters. The target sequence is shifted one character forward,
so each input position is trained to predict the next character.

### Embeddings

Each token ID is mapped to a `N_EMBED`-dimensional token embedding. A learned
position embedding is added to it so the model can distinguish positions in
the context window.

### Transformer blocks

The model applies `N_BLOCKS` Transformer blocks in sequence. Each block has two
pre-LayerNorm residual sublayers:

1. **Multi-head causal self-attention**
   - Each head independently projects the input into key, query, and value
     vectors.
   - Scaled query-key dot products are normalized with softmax to obtain
     attention weights.
   - A lower-triangular causal mask prevents a position from attending to
     future characters.
   - Head outputs are concatenated, linearly projected back to `N_EMBED`, and
     passed through dropout.
   - The attention output is added to the block input.
2. **Position-wise feed-forward network**
   - A linear layer expands the embedding width to `4 * N_EMBED`.
   - ReLU adds a non-linearity, followed by a linear layer that projects back
     to `N_EMBED`.
   - Dropout is applied, and the result is added to the attention residual.

After all blocks, a final LayerNorm is applied, followed by a linear language
model head that produces a logit for each vocabulary character at every
position.

### Current model settings

The values are constants near the top of `v2.py`:

| Setting | Value |
| --- | ---: |
| Batch size | 64 |
| Context length (`BLOCK_SIZE`) | 256 characters |
| Embedding width (`N_EMBED`) | 384 |
| Transformer blocks (`N_BLOCKS`) | 6 |
| Attention heads per block | 4 |
| Per-head key/query/value width (`HEAD_SIZE`) | 6 |
| Dropout probability | 0.2 |

Each set of head outputs is concatenated and projected to the model embedding
width. The configured head width need not equal `N_EMBED / number_of_heads`
because this implementation uses an output projection after concatenation.

### Training and generation

Training minimizes cross-entropy over next-character predictions using the
AdamW optimizer. The script runs for `MAX_ITERS` steps and estimates train and
validation loss every `EVAL_INTERVAL` steps by averaging 50 batches for each
split. Dropout is enabled during training and disabled for evaluation and
validation because the evaluation helper switches the model to evaluation
mode. After the final training step, the script does not switch the model to
evaluation mode before generation, so dropout remains enabled while sampling.

Generation starts with a single zero-valued token ID. At each step, the model
uses at most the most recent `BLOCK_SIZE` tokens, applies softmax to the final
position's logits, samples the next token from that distribution, and appends
it to the sequence.

## Project files

- `bigram.py` — embedding-based baseline language model.
- `v2.py` — multi-head causal Transformer language model.
- `gpt.ipynb`, `gpt_dev.ipynb` — notebooks used for exploring the model
  components.
- `input.txt` — local training text; not tracked by Git.
