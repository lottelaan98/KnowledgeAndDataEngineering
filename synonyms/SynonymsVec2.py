from pathlib import Path
import json
import re

import numpy as np
import fasttext
import faiss
from rdflib import Graph, Namespace, URIRef

# --------- paths ----------
BASE_DIR1 = Path(__file__).resolve().parent.parent
TTL_PATH = BASE_DIR1 / "ontology" / "databaseV6.ttl"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "BioSentVec_PubMed_MIMICIII-bigram_d700.bin"

OUT_DIR = BASE_DIR / "faiss_subset"
INDEX_PATH = OUT_DIR / "index.faiss"
META_PATH  = OUT_DIR / "symptoms_meta.json"

# OPTIONAL label mapping (if you create it later)
LABELS_PATH = BASE_DIR / "symptom_labels.json"

# --------- namespaces (MUST match your TTL) ----------
EX  = Namespace("http://example.org/med#")
WD  = Namespace("http://www.wikidata.org/entity/")
SYM = Namespace("http://example.org/med#symptom/")

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

def uri_to_key(u: str) -> str:
    """
    Convert a symptom URI into a stable key that matches your KB style:
      - WD QID:  http://www.wikidata.org/entity/Q38933 -> "Q38933"
      - sym URI: http://example.org/med#symptom/loss_of_appetite -> "sym:loss_of_appetite"
    """
    if u.startswith(str(WD)):
        return u.rsplit("/", 1)[-1]  # Qxxxx
    if u.startswith(str(SYM)):
        return "sym:" + u[len(str(SYM)):]  # loss_of_appetite
    return u  # fallback full URI

def key_to_fallback_label(key: str) -> str:
    """
    If you don't have a human label mapping yet, make a readable fallback.
    """
    if key.startswith("sym:"):
        return key.split("sym:", 1)[1].replace("_", " ")
    return key  # for QIDs, fallback is the QID itself

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load TTL
    g = Graph()
    g.parse(str(TTL_PATH), format="turtle")
    print("Total triples:", len(g))

    # Debug: check predicate counts
    preds = [EX.hasPrimarySymptom, EX.hasSecondarySymptom, EX.hasRareSymptom]
    for p in preds:
        c = sum(1 for _ in g.triples((None, p, None)))
        print(f"{p} -> {c}")

    # Extract symptom URIs
    symptom_uris = set()
    for p in preds:
        for _, _, o in g.triples((None, p, None)):
            if isinstance(o, URIRef):
                symptom_uris.add(str(o))

    symptom_uris = sorted(symptom_uris)
    print("Found symptom URIs:", len(symptom_uris))

    if not symptom_uris:
        raise RuntimeError(
            "No symptom URIs found. This usually means the EX namespace doesn't match your TTL "
            "or the predicates in the TTL differ from hasPrimary/Secondary/RareSymptom."
        )

    # Optional labels mapping
    labels = {}
    if LABELS_PATH.exists():
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        print("Loaded symptom_labels.json:", len(labels))
    else:
        print("symptom_labels.json not found (OK) -> using fallback labels")

    # Build meta + texts
    meta = []
    texts = []

    for u in symptom_uris:
        key = uri_to_key(u)  # Qxxxx or sym:xxxx
        label = labels.get(key) or key_to_fallback_label(key)
        label = normalize_text(label)

        row = len(meta)
        meta.append({
            "row": row,
            "key": key,
            "uri": u,
            "wd_qid": key if key.startswith("Q") else None,
            "text": label
        })
        texts.append(label)

    # Load model and embed
    print("Loading BioSentVec model...")
    model = fasttext.load_model(str(MODEL_PATH))
    dim = model.get_dimension()

    print("Embedding symptoms...")
    X = np.vstack([model.get_sentence_vector(t) for t in texts]).astype("float32")
    X = normalize_rows(X)

    # Build FAISS
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    # Save
    print("Saving index + meta...")
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print("Index:", INDEX_PATH)
    print("Meta :", META_PATH)

if __name__ == "__main__":
    main()
