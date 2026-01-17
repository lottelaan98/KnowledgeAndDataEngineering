# fetch_wikidata_symptom_labels.py
from pathlib import Path
import json
from typing import List, Set

import requests
from rdflib import Graph, Namespace, URIRef

BASE_DIR1 = Path(__file__).resolve().parent.parent
TTL_PATH = BASE_DIR1 / "ontology" / "databaseV6.ttl"

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "symptom_labels.json"

EX = Namespace("http://example.org/med#")
WD = Namespace("http://www.wikidata.org/entity/")

PREDICATES = ["hasPrimarySymptom", "hasSecondarySymptom", "hasRareSymptom"]

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "KDE2-symptom-label-fetcher/1.0 (contact: local-script)"
}

def chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def extract_wd_qids_from_ttl(ttl_path: Path) -> Set[str]:
    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    qids: Set[str] = set()
    for pred_local in PREDICATES:
        pred = EX[pred_local]
        for _, _, o in g.triples((None, pred, None)):
            if isinstance(o, URIRef) and str(o).startswith(str(WD)):
                qid = str(o).rsplit("/", 1)[-1]
                if qid.startswith("Q"):
                    qids.add(qid)
    return qids

def fetch_labels_for_qids(qids: List[str]) -> dict:
    """
    Returns { "Q188008": "shortness of breath", ... }
    """
    labels = {}

    # Wikidata endpoint prefers not-too-huge queries; chunk to be safe.
    for batch in chunked(qids, 200):
        values = " ".join(f"wd:{q}" for q in batch)
        sparql = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?item ?label WHERE {{
          VALUES ?item {{ {values} }}
          ?item rdfs:label ?label .
          FILTER(LANG(?label) = "en")
        }}
        """

        r = requests.get(WIKIDATA_SPARQL, params={"query": sparql}, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()

        for b in data["results"]["bindings"]:
            uri = b["item"]["value"]     # full URI
            label = b["label"]["value"]
            qid = uri.rsplit("/", 1)[-1]
            labels[qid] = label

    return labels

def main():
    qids = sorted(extract_wd_qids_from_ttl(TTL_PATH))
    print("Found Wikidata symptom QIDs:", len(qids))

    # Load existing cache (if any) so you don't refetch every time
    existing = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        print("Loaded existing symptom_labels.json:", len(existing))

    missing = [q for q in qids if q not in existing]
    print("Missing labels to fetch:", len(missing))

    if missing:
        new_labels = fetch_labels_for_qids(missing)
        existing.update(new_labels)

    OUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved labels to:", OUT_PATH)

if __name__ == "__main__":
    main()
