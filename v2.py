"""A character-level language model with multi-head self-attention."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DATA_PATH = Path("input.txt")
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EMBED = 384
HEAD_SIZE = 6
N_BLOCKS = 6
DROPOUT = 0.2


class SingleHeadAttention(nn.Module):
    def __init__(self, head_size=HEAD_SIZE):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.scale = head_size ** -0.5

    def forward(self, x):
        _, time, _ = x.shape
        keys = self.key(x)
        queries = self.query(x)
        weights = queries @ keys.transpose(-2, -1) * self.scale
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    """A multi-head self-attention mechanism."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(head_size) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(num_heads * head_size, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.projection(torch.cat([head(x) for head in self.heads], dim=-1))
        return self.dropout(x)


class FeedForward(nn.Module):
    """A position-wise feed-forward network."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    """A transformer block with multi-head attention and a feed-forward layer."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward()
        self.layer_norm1 = nn.LayerNorm(N_EMBED)
        self.layer_norm2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        return x + self.feed_forward(self.layer_norm2(x))


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(
            *(Block(num_heads=4, head_size=HEAD_SIZE) for _ in range(N_BLOCKS))
        )
        self.final_layer_norm = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, idx, targets=None):
        _, time = idx.shape
        token_embeddings = self.token_embedding(idx)
        positions = torch.arange(time, device=idx.device)
        x = token_embeddings + self.position_embedding_table(positions)
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            context = idx[:, -BLOCK_SIZE:]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def main():
    text = DATA_PATH.read_text(encoding="utf-8")
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    def get_batch(source):
        if len(source) <= BLOCK_SIZE:
            raise ValueError("The dataset must contain more than BLOCK_SIZE characters.")
        starts = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([source[i : i + BLOCK_SIZE] for i in starts])
        y = torch.stack([source[i + 1 : i + BLOCK_SIZE + 1] for i in starts])
        return x.to(DEVICE), y.to(DEVICE)

    model = BigramLanguageModel(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        for name, source in (("train", train_data), ("val", val_data)):
            batch_losses = []
            for _ in range(50):
                x, y = get_batch(source)
                _, loss = model(x, y)
                batch_losses.append(loss.item())
            losses[name] = sum(batch_losses) / len(batch_losses)
        model.train()
        return losses

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0:
            print(step, estimate_loss())
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    start = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(start, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
