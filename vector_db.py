# vector_db.py
import json, pathlib, numpy as np
from typing import List

# 🔸 Replace with your preferred model, e.g. OpenAI, Sentence‑Transformers, etc.
from sentence_transformers import SentenceTransformer
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

ROOT = pathlib.Path(__file__).parent

def _load_terms(fname: str) -> List[str]:
    return (ROOT / "concepts" / fname).read_text().splitlines()

# Pre‑embed your concept dictionaries once at startup
SAFE_VECS   = _MODEL.encode(_load_terms("safe_terms.txt"),   normalize_embeddings=True)
SPIRAL_VECS = _MODEL.encode(_load_terms("spiral_terms.txt"), normalize_embeddings=True)
NULL_VECS   = _MODEL.encode(_load_terms("null_terms.txt"),   normalize_embeddings=True)

# Optional polarity pairs for antivector checks
POLARITY    = json.loads((ROOT / "concepts" / "polarity_pairs.json").read_text())

# General helpers -----------------------------------------------------------
def embed(texts: List[str]) -> np.ndarray:
    return _MODEL.encode(texts, normalize_embeddings=True)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(a, b.T)

def max_similarity(vec: np.ndarray, bank: np.ndarray) -> float:
    return float(np.max(cosine_similarity(vec[None, :], bank)))
