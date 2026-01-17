# extract_symptoms.py
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Dict, Tuple

_WS = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\-]", " ", s)   # remove punctuation, keep hyphens
    s = _WS.sub(" ", s)
    return s

def load_symptom_phrases(meta_path: Path) -> List[str]:
    """
    Reads your KB symptom list (from symptoms_meta.json produced offline)
    and returns the normalized symptom phrases.
    Expected format: [{"id": 0, "text": "cough"}, ...]
    """
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return [normalize_text(m["text"]) for m in meta if "text" in m]

def extract_symptoms_from_text(
    patient_text: str,
    symptom_phrases: List[str],
    *,
    max_matches: int = 20
) -> List[str]:
    """
    Very fast baseline extractor:
    - normalizes text
    - finds occurrences of known symptom phrases
    - returns unique matches, preferring longer phrases first
    """
    text = normalize_text(patient_text)

    # Prefer longer matches so "shortness of breath" wins over "breath"
    phrases = sorted(set(symptom_phrases), key=len, reverse=True)

    found: List[str] = []
    used_spans: List[Tuple[int, int]] = []

    for p in phrases:
        if not p:
            continue

        # word-boundary-ish match (works decently for phrases)
        # e.g. "dry cough" should not match "dry coughing" unless you want it to
        pattern = r"(?:^|[\s])" + re.escape(p) + r"(?:$|[\s])"
        for m in re.finditer(pattern, text):
            start, end = m.span()

            # prevent heavy overlap duplicates
            overlap = any(not (end <= s or start >= e) for s, e in used_spans)
            if overlap:
                continue

            used_spans.append((start, end))
            found.append(p)

            if len(found) >= max_matches:
                return found

    # If nothing matched, fall back to splitting into short candidate chunks
    # (so canonicalization can still try to map them)
    if not found:
        chunks = [c.strip() for c in re.split(r"[.;,\n]| and | but | or ", text) if c.strip()]
        # Keep short-ish chunks
        found = [c for c in chunks if 2 <= len(c) <= 60][:max_matches]

    return found


if __name__ == "__main__":
    # Example usage:
    BASE_DIR = Path(__file__).resolve().parent
    META_PATH = BASE_DIR / "faiss_subset" / "symptoms_meta.json"

    phrases = load_symptom_phrases(META_PATH)

    patient = "I've had a dry cough for 3 days and trouble breathing at night. Also chest tightness."
    extracted = extract_symptoms_from_text(patient, phrases)

    print("Extracted symptom phrases:")
    for s in extracted:
        print("-", s)
