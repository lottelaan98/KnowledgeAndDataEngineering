# symptom_match_pipeline.py
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Dict

import numpy as np
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_WS = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\-]", " ", s)   # keep hyphens
    s = _WS.sub(" ", s)
    return s

def tokenize(s: str) -> List[str]:
    return _WORD_RE.findall(normalize_text(s))

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?]+\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]

def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def candidate_phrases(text: str, max_phrases: int = 300) -> List[str]:
    """
    Build short symptom-like candidates from input.
    This fixes the 'whole sentence embedding is too noisy' problem.

    Produces:
      - windows around anchor words (rash/itchy/scaly/patches/etc.)
      - ngrams (1-3) to capture phrases like "skin rash", "dry scaly patches"
    """
    t = normalize_text(text)
    toks = tokenize(t)
    if not toks:
        return []

    anchors = {
        "rash", "itch", "itchy", "itching", "red", "scaly", "dry",
        "patch", "patches", "spots", "sores", "blister", "vesicle"
    }

    # 1) keyword windows (more precise than pure ngrams)
    windows = []
    for i, w in enumerate(toks):
        if w in anchors:
            left = max(0, i - 2)
            right = min(len(toks), i + 3)
            windows.append(" ".join(toks[left:right]))

    # 2) ngrams (1-3)
    ngrams = []
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            ngrams.append(" ".join(toks[i:i+n]))

    # de-dup, preserve order
    seen = set()
    out: List[str] = []
    for p in windows + ngrams:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= max_phrases:
            break

    return out


class HybridSymptomMatcher:
    """
    TF-IDF shortlist + fastText rerank WITHOUT FAISS.

    Startup (1x):
      - Load symptoms_meta.json (URI/key/label)
      - Build TF-IDF over labels
      - Precompute label embeddings into matrix E (cached)

    Per query:
      - Generate candidate phrases from user input
      - TF-IDF shortlist (top N)
      - Embed candidate phrase
      - cosine via dot product: scores = E_short @ q
      - keep top-k above threshold
    """

    def __init__(
        self,
        model_path: Path,
        meta_path: Path,
        cache_dir: Path,
        cache_dtype: str = "float32",
    ):
        self.model = fasttext.load_model(str(model_path))
        self.dim = self.model.get_dimension()

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.labels = [m["text"] for m in self.meta]   # already normalized in your builder
        self.keys   = [m["key"] for m in self.meta]
        self.uris   = [m["uri"] for m in self.meta]
        self.qids   = [m.get("wd_qid") for m in self.meta]

        # TF-IDF on labels
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        self.X = self.vectorizer.fit_transform(self.labels)

        # Cache label embedding matrix E
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.E_path = cache_dir / "E_labels.npy"

        if self.E_path.exists():
            self.E = np.load(self.E_path)
        else:
            print("Building label embedding matrix E (1x)...")
            E = np.vstack([self.embed_phrase(lbl) for lbl in self.labels]).astype("float32")
            if cache_dtype == "float16":
                E = E.astype("float16")
            np.save(self.E_path, E)
            self.E = E

    def embed_phrase(self, s: str) -> np.ndarray:
        # For BioSentVec/fastText: get_sentence_vector is convenient
        v = np.array(self.model.get_sentence_vector(normalize_text(s)), dtype="float32")
        v = v.reshape(1, -1)
        v = normalize_rows(v)
        return v[0]

    def tfidf_shortlist(self, text: str, top_n: int = 40) -> List[int]:
        qv = self.vectorizer.transform([normalize_text(text)])
        sims = cosine_similarity(qv, self.X).ravel()

        if top_n >= len(sims):
            return list(np.argsort(-sims))

        idx = np.argpartition(-sims, top_n)[:top_n]
        idx = idx[np.argsort(-sims[idx])]
        return idx.tolist()

    def rerank_embeddings(
        self,
        text: str,
        candidate_idxs: List[int],
        top_k: int = 5,
        threshold: float = 0.50,   # NOTE: lowered default threshold
    ) -> List[Dict]:
        q = self.embed_phrase(text)
        if np.allclose(q, 0):
            return []

        candE = self.E[candidate_idxs].astype(np.float32)
        scores = candE @ q  # cosine because normalized
        order = np.argsort(-scores)

        out = []
        for j in order[:top_k]:
            score = float(scores[j])
            if score < threshold:
                continue
            idx = candidate_idxs[j]
            out.append({
                "score": score,
                "key": self.keys[idx],
                "uri": self.uris[idx],
                "wd_qid": self.qids[idx],
                "label": self.labels[idx],
                "matched_from": text,
            })
        return out

    def match_text(
        self,
        patient_text: str,
        tfidf_top_n: int = 40,
        emb_top_k: int = 5,
        emb_threshold: float = 0.50,  # NOTE: lowered default threshold
        per_input_limit: int = 50,
    ) -> List[Dict]:
        """
        Candidate phrases → TF-IDF shortlist → embedding rerank → aggregate best per key.
        """
        best: Dict[str, Dict] = {}

        phrases = candidate_phrases(patient_text)
        # Optional: also include whole sentences, but phrases usually work better
        phrases = phrases + split_sentences(patient_text)

        for p in phrases[:per_input_limit]:
            cand = self.tfidf_shortlist(p, top_n=tfidf_top_n)
            matches = self.rerank_embeddings(p, cand, top_k=emb_top_k, threshold=emb_threshold)

            for m in matches:
                key = m["key"]
                if (key not in best) or (m["score"] > best[key]["score"]):
                    best[key] = m

        out = list(best.values())
        out.sort(key=lambda x: x["score"], reverse=True)
        return out


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent

    matcher = HybridSymptomMatcher(
        model_path=BASE / "BioSentVec_PubMed_MIMICIII-bigram_d700.bin",
        meta_path=BASE / "faiss_subset" / "symptoms_meta.json",
        cache_dir=BASE / "cache_runtime",
        cache_dtype="float32",  # change to float16 to halve size
    )

    text = (
        "I have had a high fever for over a week now that doesn’t seem to go down. I feel extremely tired and weak, with a constant headache. I’ve lost my appetite and often feel nauseous, sometimes even vomiting."
    )

    results = matcher.match_text(
        text,
        tfidf_top_n=40,
        emb_top_k=5,
        emb_threshold=0.50,   # lowered
    )

    print("\nInput:\n", text)
    print("\nMatched ontology symptoms:")
    if not results:
        print("(none)")
    else:
        for r in results[:10]:
            print(
                f"- {r['label']:30s} score={r['score']:.3f} "
                f"key={r['key']} uri={r['uri']}  (from: '{r['matched_from']}')"
            )
