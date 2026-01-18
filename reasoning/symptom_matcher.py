import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import rdflib
from rdflib import Graph, Namespace, RDF
from rdflib.namespace import SKOS, RDFS

# Optional: spaCy
try:
    import spacy
except Exception:
    spacy = None


EX = Namespace("http://example.org/med#")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_symptom_labels(graph: Graph) -> List[Tuple[rdflib.term.Identifier, str]]:
    """Return list of (symptom_uri, label) for all symptoms in KG."""
    symptom_types = {EX.Symptom}
    for subclass in graph.subjects(RDFS.subClassOf, EX.Symptom):
        symptom_types.add(subclass)

    out = []
    for symptom_type in symptom_types:
        for s in graph.subjects(RDF.type, symptom_type):
            label = None
            for p in (SKOS.prefLabel, RDFS.label):
                for l in graph.objects(s, p):
                    if not getattr(l, "language", None) or l.language == "en":
                        label = str(l)
                        break
                if label:
                    break
            if label:
                out.append((s, label))
    return out


def extract_phrases_simple(text: str) -> List[str]:
    """
    Minimal phrase extraction that works even without spaCy:
    - unigrams + some bigrams
    """
    t = normalize(text)
    tokens = t.split()
    phrases = set(tokens)
    for i in range(len(tokens) - 1):
        phrases.add(tokens[i] + " " + tokens[i + 1])
    return sorted(phrases)


def extract_phrases_spacy(text: str, model: str = "en_core_web_sm") -> List[str]:
    """
    Phrase extraction with spaCy if available.
    If model not installed, fallback to simple extractor.
    """
    if spacy is None:
        return extract_phrases_simple(text)

    try:
        nlp = spacy.load(model)
    except Exception:
        return extract_phrases_simple(text)

    doc = nlp(text)
    phrases = set()

    # noun chunks need parser; guard
    if doc.has_annotation("DEP"):
        for chunk in doc.noun_chunks:
            phrases.add(normalize(chunk.text))

    # add lemmas of nouns/adjs
    for token in doc:
        if token.pos_ in {"NOUN", "ADJ"} and not token.is_stop:
            phrases.add(normalize(token.lemma_))

    # also raw tokens
    for token in doc:
        if token.is_alpha and not token.is_stop:
            phrases.add(normalize(token.text))

    phrases = {p for p in phrases if p and len(p) >= 3}
    return sorted(phrases)


def build_tfidf_vectors(texts: List[str]):
    """
    TF-IDF vectors (cosine similarity) using scikit-learn.
    If sklearn not installed, raise helpful error.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for cosine similarity TF-IDF.\n"
            "Install with: pip install scikit-learn"
        ) from e

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def match_symptoms(graph: Graph, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Compare user input against KG symptom labels using cosine similarity (TF-IDF).
    Returns top_k best matches: [{"uri":..., "label":..., "score":...}, ...]
    """
    # Extract phrases from user
    phrases = extract_phrases_spacy(text)  # auto-fallback if model missing
    user_doc = " ".join(phrases) if phrases else normalize(text)

    # Get symptom labels
    symptom_items = get_symptom_labels(graph)
    if not symptom_items:
        return []

    labels = [lbl for _, lbl in symptom_items]
    corpus = [user_doc] + labels

    _, X = build_tfidf_vectors(corpus)

    # cosine similarity between user vector and all symptom label vectors
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for cosine similarity.\n"
            "Install with: pip install scikit-learn"
        ) from e

    sims = cosine_similarity(X[0], X[1:]).flatten()

    results = []
    for (uri, lbl), score in zip(symptom_items, sims):
        results.append({"uri": str(uri), "label": lbl, "score": float(score)})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]

def symptoms_above_threshold(
    matches: List[Dict[str, Any]],
    threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Filter symptom match results to only those with score > threshold.
    Returns the same dict structure: {"uri":..., "label":..., "score":...}
    """
    return [m for m in matches if float(m.get("score", 0.0)) > threshold]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ttl",
        default="ontology/databaseV7.ttl",
        help="TTL file path (default: ontology/databaseV7.ttl)",
    )
    ap.add_argument(
        "--text",
        default=(
            "I have fever, cough, and shortness of breath. I also have chest pain and I feel very tired and weak."
        ),
        help="User input text",
    )
    ap.add_argument("--topk", type=int, default=5, help="Top K symptom matches")

    args = ap.parse_args()

    ttl_path = Path(args.ttl)
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path.resolve()}")

    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    g.bind("ex", EX)
    g.bind("skos", SKOS)

    results = match_symptoms(g, args.text, top_k=args.topk)

    print("=== INPUT ===")
    print(args.text)
    print("\n=== BEST SYMPTOM MATCHES (cosine TF-IDF) ===")

    if not results:
        print("(no symptom labels found in KG)")
        return

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['label']}")
        print(f"   score = {r['score']:.3f}")
        print(f"   uri   = {r['uri']}")
        print()


if __name__ == "__main__":
    main()
