"""PyTorch ticket classifier that runs on an NVIDIA GPU when one is available.

Same task and synthetic data as the scikit-learn service (inference/app/data.py),
but as a neural model (EmbeddingBag + MLP) so it can be moved to CUDA.
Device placement is explicit: nothing silently claims to be on a GPU.
"""

import random
import re
from collections import Counter
from itertools import pairwise

import torch
from torch import nn

from app.data import LABELS, generate  # from inference/; put it on PYTHONPATH (see pyproject.toml)

TOKEN_RE = re.compile(r"[a-z0-9']+")
PAD, UNK = 0, 1


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    # Unigrams + bigrams, mirroring the TF-IDF baseline's ngram_range=(1, 2).
    return [*tokens, *(f"{a}_{b}" for a, b in pairwise(tokens))]


class Vocab:
    def __init__(self, texts: list[str], min_freq: int = 2):
        counts = Counter(t for text in texts for t in tokenize(text))
        self.itos = ["<pad>", "<unk>", *sorted(t for t, c in counts.items() if c >= min_freq)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat token ids + offsets, the input format nn.EmbeddingBag expects."""
        ids: list[int] = []
        offsets: list[int] = []
        for text in texts:
            offsets.append(len(ids))
            ids.extend(self.stoi.get(t, UNK) for t in tokenize(text) or ["<unk>"])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class TicketClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden: int = 1024, n_classes: int = len(LABELS)):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=PAD)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(ids, offsets))


Split = tuple[list[str], list[int]]


def split(n: int = 4000, seed: int = 42, test_frac: float = 0.2) -> tuple[Split, Split]:
    """Shuffled split before building the vocabulary, so test tokens never leak into it."""
    texts, labels = generate(n=n, seed=seed)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(n * (1 - test_frac))

    def pick(ids: list[int]) -> Split:
        return [texts[i] for i in ids], [LABELS.index(labels[i]) for i in ids]

    return pick(idx[:cut]), pick(idx[cut:])


def train(
    device: torch.device, epochs: int = 8, batch_size: int = 64, seed: int = 42
) -> tuple[TicketClassifier, Vocab, dict]:
    torch.manual_seed(seed)
    (x_train, y_train), (x_test, y_test) = split(seed=seed)
    vocab = Vocab(x_train)
    model = TicketClassifier(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    order = list(range(len(x_train)))
    rng = random.Random(seed)
    for _ in range(epochs):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            ids, offsets = vocab.encode_batch([x_train[i] for i in batch])
            target = torch.tensor([y_train[i] for i in batch], device=device)
            opt.zero_grad()
            loss = loss_fn(model(ids.to(device), offsets.to(device)), target)
            loss.backward()
            opt.step()

    return model, vocab, evaluate(model, vocab, x_test, y_test, device)


@torch.inference_mode()
def evaluate(model: nn.Module, vocab: Vocab, texts: list[str], labels: list[int], device: torch.device) -> dict:
    model.eval()
    ids, offsets = vocab.encode_batch(texts)
    pred = model(ids.to(device), offsets.to(device)).argmax(1).cpu()
    y = torch.tensor(labels)
    f1s = []
    for c in range(len(LABELS)):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "accuracy": round((pred == y).float().mean().item(), 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "n_test": len(labels),
    }
