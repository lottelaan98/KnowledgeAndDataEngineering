# canonicalize_symptoms.py
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import fasttext
import faiss

_WS = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = _WS.sub(" ", s)
    return s

def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

class SymptomCanonicalizer:
    def __init__(
        self,
        model_path: Path,
        index_path: Path,
        meta_path: Path,
        *,
        accept_threshold: float = 0.62,
        ambiguity_delta: float = 0.08
    ):
        """
        accept_threshold: minimum cosine similarity to accept top1 match
        ambiguity_delta: if top1 - top2 < ambiguity_delta, mark as ambiguous
        """
        self.accept_threshold = accept_threshold
        self.ambiguity_delta = ambiguity_delta

        self.model = fasttext.load_model(str(model_path))
        self.index = faiss.read_index(str(index_path))
        self.meta: List[Dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))

        # Sanity check: index size should match meta length
        if self.index.ntotal != len(self.meta):
            raise ValueError(
                f"Index size (ntotal={self.index.ntotal}) != meta length ({len(self.meta)}). "
                "Make sure meta and index were built together."
            )

    def _embed(self, text: str) -> np.ndarray:
        v = np.array(self.model.get_sentence_vector(text), dtype="float32").reshape(1, -1)
        v = normalize_rows(v)
        return v

    def canonicalize_one(self, phrase: str, *, k: int = 2) -> Dict[str, Any]:
        """
        Returns:
          {
            "input": original phrase,
            "normalized": normalized phrase,
            "accepted": bool,
            "match": {"id": int, "text": str, ...} | None,
            "score": float | None,
            "ambiguous": bool,
            "candidates": [{"id": int, "text": str, "score": float}, ...]  # length=k
          }
        """
        original = phrase
        phrase = normalize_text(phrase)
        if not phrase:
            return {
                "input": original,
                "normalized": phrase,
                "accepted": False,
                "match": None,
                "score": None,
                "ambiguous": False,
                "candidates": []
            }

        q = self._embed(phrase)
        D, I = self.index.search(q, k)

        candidates = []
        for rank in range(k):
            idx = int(I[0][rank])
            score = float(D[0][rank])
            m = dict(self.meta[idx])  # copy
            candidates.append({"id": idx, "text": m.get("text"), "score": score})

        top1 = candidates[0]
        top1_score = top1["score"]

        ambiguous = False
        if k >= 2:
            top2_score = candidates[1]["score"]
            ambiguous = (top1_score - top2_score) < self.ambiguity_delta

        accepted = top1_score >= self.accept_threshold and not ambiguous

        match = dict(self.meta[top1["id"]]) if accepted else None

        return {
            "input": original,
            "normalized": phrase,
            "accepted": accepted,
            "match": match,
            "score": top1_score if accepted else None,
            "ambiguous": ambiguous,
            "candidates": candidates
        }

    def canonicalize_many(self, phrases: List[str], *, k: int = 2) -> List[Dict[str, Any]]:
        return [self.canonicalize_one(p, k=k) for p in phrases]


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    MODEL_PATH = BASE_DIR / "BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
    INDEX_PATH = BASE_DIR / "faiss_subset" / "index.faiss"
    META_PATH  = BASE_DIR / "faiss_subset" / "symptoms_meta.json"

    canon = SymptomCanonicalizer(
        model_path=MODEL_PATH,
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        accept_threshold=0.62,
        ambiguity_delta=0.08
    )

    test_phrases = [
        "trouble breathing",
        "tight chest",
        "dry coughing at night",
        "i feel woozy and light headed"
    ]

    results = canon.canonicalize_many(test_phrases, k=2)
    for r in results:
        print("\nInput:", r["input"])
        print("Accepted:", r["accepted"], "Ambiguous:", r["ambiguous"])
        print("Top candidates:")
        for c in r["candidates"]:
            print(f"  {c['score']:.4f}\t{c['text']}")
        if r["accepted"]:
            print("=> Canonical:", r["match"]["text"], "score:", r["score"])
