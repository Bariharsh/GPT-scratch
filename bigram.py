"""A small character-level bigram language model."""

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


DATA_PATH = Path("input.txt")
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 32


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

	class BigramLanguageModel(nn.Module):
		def __init__(self):
			super().__init__()
			self.token_embedding = nn.Embedding(vocab_size, n_embed)
			self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embed)
			self.lm_head = nn.Linear(n_embed, vocab_size)

		def forward(self, idx, targets=None):
			B,T = idx.shape
			tok_emb = self.token_embedding(idx) # (B,T,n_embed)
			pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,n_embed)
			x = tok_emb + pos_emb # (B,T,n_embed)
			logits = self.lm_head(x) #(B,T,vocab_size)
			loss = None
			if targets is not None:
				loss = F.cross_entropy(
					logits.reshape(-1, vocab_size), targets.reshape(-1)
				)
			return logits, loss

		@torch.no_grad()
		def generate(self, idx, max_new_tokens):
			for _ in range(max_new_tokens):
				logits, _ = self(idx[:, -1:])
				probabilities = F.softmax(logits[:, -1, :], dim=-1)
				next_token = torch.multinomial(probabilities, num_samples=1)
				idx = torch.cat((idx, next_token), dim=1)
			return idx

	model = BigramLanguageModel().to(DEVICE)
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
