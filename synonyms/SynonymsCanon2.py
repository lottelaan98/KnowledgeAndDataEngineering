from pathlib import Path
import json
import re
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
    def __init__(self, model_path: Path, index_path: Path, meta_path: Path,
                 accept_threshold: float = 0.62, ambiguity_delta: float = 0.08):
        self.accept_threshold = accept_threshold
        self.ambiguity_delta = ambiguity_delta

        self.model = fasttext.load_model(str(model_path))
        self.index = faiss.read_index(str(index_path))
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if self.index.ntotal != len(self.meta):
            raise ValueError("FAISS index and meta length do not match.")

    def canonicalize_one(self, phrase: str, k: int = 2):
        phrase_norm = normalize_text(phrase)
        if not phrase_norm:
            return {"input": phrase, "accepted": False, "match": None}

        q = np.array(self.model.get_sentence_vector(phrase_norm), dtype="float32").reshape(1, -1)
        q = normalize_rows(q)

        D, I = self.index.search(q, k)

        cand = []
        for r in range(k):
            row = int(I[0][r])
            score = float(D[0][r])
            m = self.meta[row]
            cand.append({"row": row, "score": score, "key": m["key"], "wd_qid": m["wd_qid"], "text": m["text"]})

        top1 = cand[0]
        ambiguous = False
        if k >= 2:
            ambiguous = (cand[0]["score"] - cand[1]["score"]) < self.ambiguity_delta

        accepted = (top1["score"] >= self.accept_threshold) and (not ambiguous)

        return {
            "input": phrase,
            "normalized": phrase_norm,
            "accepted": accepted,
            "ambiguous": ambiguous,
            "candidates": cand,
            "match": top1 if accepted else None
        }

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    canon = SymptomCanonicalizer(
        model_path=BASE_DIR / "BioSentVec_PubMed_MIMICIII-bigram_d700.bin",
        index_path=BASE_DIR / "faiss_subset" / "index.faiss",
        meta_path=BASE_DIR / "faiss_subset" / "symptoms_meta.json",
    )

    for s in ["trouble breathing", "tight chest", "itchy eyes"]:
        r = canon.canonicalize_one(s, k=2)
        print("\nInput:", s)
        print("Accepted:", r["accepted"], "Ambiguous:", r["ambiguous"])
        if r["match"]:
            print("=> Matched text:", r["match"]["text"])
            print("=> Matched wd_qid:", r["match"]["wd_qid"])  # THIS matches TTL QIDs
