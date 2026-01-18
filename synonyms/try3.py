# symptom_match_pipeline.py
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Dict
import spacy
nlp = spacy.load("en_core_web_sm")

import numpy as np
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_WS = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = _WS.sub(" ", s)

    # tiny morphology normalization (NOT a synonym DB)
    s = s.replace("itchy", "itch")

    return s
