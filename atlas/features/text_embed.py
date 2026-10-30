"""Yangilik matnini vektorga aylantirish.

Asosiy: FinBERT (ProsusAI/finbert) — MUZLATILGAN. Faqat embedding va ton (pos-neg) olinadi,
hech qanday parametr o'qitilmaydi. Keyin PCA bilan 768 -> 64 o'lchamga tushiriladi
(PCA faqat holdoutdan oldingi ma'lumotda fit qilinadi).

Zaxira: HashingEmbedder — internet/GPU bo'lmaganda test va demo uchun.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class FinBERTEmbedder:
    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval().to(self.device)
        for p in self.model.parameters():  # muzlatish
            p.requires_grad_(False)
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.pos_i, self.neg_i = labels["positive"], labels["negative"]
        self.batch_size = batch_size
        self.dim = self.model.config.hidden_size

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        embs, tones = [], []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
                out = self.model(**enc)
                h = out.hidden_states[-1]
                m = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
                p = out.logits.softmax(-1)
                embs.append(pooled.float().cpu().numpy())
                tones.append((p[:, self.pos_i] - p[:, self.neg_i]).float().cpu().numpy())
                if (i // self.batch_size) % 50 == 0:
                    log.info("FinBERT: %d/%d", i + len(batch), len(texts))
        return np.concatenate(embs).astype("float32"), np.concatenate(tones).astype("float32")


class HashingEmbedder:
    """Tez zaxira: so'z n-gramlari -> hashing -> tasodifiy proyeksiya. Ton — leksik lug'at."""
    name = "hashing"

    def __init__(self, dim: int = 128, seed: int = 0, n_features: int = 2**15):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=True, norm="l2")
        # seed'dan deterministik proyeksiya -> servisda ham aynan bir xil vektorlar
        rng = np.random.default_rng(seed)
        self.proj = (rng.standard_normal((n_features, dim)) / np.sqrt(dim)).astype("float32")
        self.dim = dim

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        from .tagging import lexical_tone

        X = self.vec.transform(texts)
        E = np.asarray(X @ self.proj, dtype="float32")
        tone = np.array([lexical_tone(t) for t in texts], dtype="float32")
        return E, tone


def get_embedder(kind: str = "auto"):
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("auto", "finbert"):
        try:
            return FinBERTEmbedder()
        except Exception as e:
            if kind == "finbert":
                raise
            log.warning("FinBERT yuklanmadi (%s) -> HashingEmbedder ishlatiladi", e)
    return HashingEmbedder()


class PCAReducer:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.mean = None
        self.comp = None

    def fit(self, X: np.ndarray) -> "PCAReducer":
        self.mean = X.mean(0)
        Xc = X - self.mean
        if len(Xc) > 200_000:
            Xc = Xc[np.random.default_rng(0).choice(len(Xc), 200_000, replace=False)]
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        self.comp = vt[: self.dim].T.astype("float32")
        sd = s[: self.dim] / np.sqrt(len(Xc))
        # Juda kichik dispersiyali komponentlarni "oqartirish" shovqinni portlatadi -> pastki chegara
        self.scale = np.maximum(sd, 0.05 * sd[0] + 1e-8).astype("float32")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean) @ self.comp / self.scale, -10, 10).astype("float32")

    def state(self) -> dict:
        return {"mean": self.mean, "comp": self.comp, "scale": self.scale}

    @classmethod
    def from_state(cls, st: dict) -> "PCAReducer":
        r = cls(st["comp"].shape[1])
        r.mean, r.comp, r.scale = st["mean"], st["comp"], st["scale"]
        return r
