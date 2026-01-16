# LLM_batch.py
# Run:
#   python LLM_batch.py
#
# Requirements:
#   pip install requests rdflib
#
# Prereqs:
#   Ollama running locally: http://localhost:11434
#   Model pulled: ollama pull llama3

from __future__ import annotations

import html
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode

import requests
from rdflib import Graph


# ----------------------------
# Paths / Config
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "ontology" / "databaseV6.ttl"

OUT_PATH = BASE_DIR / "ontology" / "disease_summaries.json"   # change if you want
CHECKPOINT_PATH = BASE_DIR / "ontology" / "disease_summaries_checkpoint.json"

MEDLINEPLUS_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

REQUEST_SLEEP_SECONDS = 0.2     # be polite; increase if you get throttled
OLLAMA_TIMEOUT_SECONDS = 240    # large summaries can take longer

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# ----------------------------
def clean_medline_text(s: str) -> str:
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def fetch_medlineplus_full_summary(term: str, db: str = "healthTopics") -> dict | None:
    params = {"db": db, "term": term, "rettype": "brief", "retmax": 5}
    url = f"{MEDLINEPLUS_BASE_URL}?{urlencode(params)}"

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    doc = root.find("./list/document")
    if doc is None:
        return None

    title = None
    full_summary = None

    for c in doc.findall("./content"):
        name = c.attrib.get("name", "")
        raw = "".join(c.itertext())
        if name == "title":
            title = clean_medline_text(raw)
        elif name in ("FullSummary", "fullSummary"):
            full_summary = clean_medline_text(raw)

    return {
        "query": term,
        "title": title,
        "url": doc.attrib.get("url"),
        "full_summary": full_summary or "",
    }


def ollama_generate(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json().get("response", "")


def extract_json(text: str) -> dict:
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            t = m.group(0).strip()
    return json.loads(t)


def load_diseases_from_ttl(ttl_path: Path) -> list[dict]:
    g = Graph()
    g.parse(ttl_path, format="turtle")

    q = """
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX ex:   <http://example.org/med#>

    SELECT DISTINCT ?disease ?name
    WHERE {
        ?disease rdf:type ex:Disease .
        ?disease skos:prefLabel ?name .
        FILTER (lang(?name) = "en")
    }
    ORDER BY ?name
    """

    diseases: list[dict] = []
    for row in g.query(q):
        disease_uri = str(row.disease)
        q_id = disease_uri.split("/")[-1]
        name = str(row.name).strip()

        diseases.append({"q_id": q_id, "name": name})

    return diseases


def build_prompt(mp: dict, q_id: str, disease_name: str) -> str:
    # IMPORTANT: Put q_id into the disease line as you requested.
    disease_line = mp["title"] or disease_name

    return f"""
You are a medical summarization assistant.
Use ONLY the provided MedlinePlus summary text. Do NOT invent facts.
If something is not mentioned in the summary, say "Not specified in the source."
Write for a general audience.

Return STRICT JSON ONLY (no markdown, no backticks) with exactly these keys:
- explanation_100_words_max: string (<= 100 words)
- symptoms: array of strings
- treatment_options: string (<= 100 words)
- see_a_doctor: object with keys:
    - recommended: boolean
    - urgency: one of ["emergency", "urgent", "routine", "unclear"]
    - guidance: string

DISEASE/CONDITION: {disease_line} (q_id: {q_id})

MEDLINEPLUS FULL SUMMARY (SOURCE TEXT):
{mp["full_summary"]}

Now output the JSON.
""".strip()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    diseases = load_diseases_from_ttl(DB_PATH)
    print(f"Loaded {len(diseases)} diseases from: {DB_PATH}")

    # Resume if checkpoint exists
    results_by_qid: dict[str, dict] = {}
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                results_by_qid = json.load(f)
            print(f"Resuming from checkpoint: {CHECKPOINT_PATH} ({len(results_by_qid)} done)")
        except Exception:
            results_by_qid = {}

    for i, d in enumerate(diseases, start=1):
        q_id = d["q_id"]
        name = d["name"]

        if q_id in results_by_qid:
            continue  # already processed

        print(f"[{i}/{len(diseases)}] {name} ({q_id})")

        # Fetch MedlinePlus
        try:
            mp = fetch_medlineplus_full_summary(name)
        except Exception as e:
            results_by_qid[q_id] = {
                "q_id": q_id,
                "disease_name": name,
                "error": "MedlinePlus fetch failed",
                "details": str(e),
            }
            save_json(CHECKPOINT_PATH, results_by_qid)
            continue

        if not mp or not mp.get("full_summary"):
            results_by_qid[q_id] = {
                "q_id": q_id,
                "disease_name": name,
                "error": "No MedlinePlus FullSummary found",
                "source_url": mp.get("url") if mp else None,
            }
            save_json(CHECKPOINT_PATH, results_by_qid)
            time.sleep(REQUEST_SLEEP_SECONDS)
            continue

        # Build prompt and call Ollama
        prompt = build_prompt(mp, q_id=q_id, disease_name=name)
        try:
            raw = ollama_generate(prompt)
            chunks = extract_json(raw)
        except Exception as e:
            results_by_qid[q_id] = {
                "q_id": q_id,
                "disease_name": name,
                "error": "Failed to parse model output as JSON",
                "details": str(e),
                "model_output": raw if "raw" in locals() else None,
                "source_title": mp.get("title") or name,
                "source_url": mp.get("url"),
            }
            save_json(CHECKPOINT_PATH, results_by_qid)
            time.sleep(REQUEST_SLEEP_SECONDS)
            continue

        # Store EXACT keys + add provenance outside (if you want to keep strict only, remove provenance fields)
        # If you need the output JSON to be ONLY the strict keys, put provenance at top-level wrapper instead.
        results_by_qid[q_id] = {
            "q_id": q_id,
            "disease_name": name,
            "source_title": mp.get("title") or name,
            "source_url": mp.get("url"),
            "summary": chunks,  # <-- contains ONLY your strict keys if the model follows instructions
        }

        save_json(CHECKPOINT_PATH, results_by_qid)
        time.sleep(REQUEST_SLEEP_SECONDS)

    # Final save
    save_json(OUT_PATH, results_by_qid)
    print(f"Done. Saved results to: {OUT_PATH}")


if __name__ == "__main__":
    main()
