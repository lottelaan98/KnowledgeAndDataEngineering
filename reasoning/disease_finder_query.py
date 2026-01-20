"""
Query-based disease finder (output formatted like the pure-Python reference).

Output format:
Input symptoms: [...]
1. DiseaseName
   Similarity: xx.xx%
   Matched: symptom1, symptom2
"""

import sys
from pathlib import Path
from typing import List, Optional

from rdflib import Graph

# Make project root importable so "from querying import scriptV3" works
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from querying import scriptV3  


def load_graph(ttl_path: str) -> Graph:
    g = Graph()
    g.parse(ttl_path, format="turtle")
    return g


def _symptom_ids_or_passthrough(symptoms: List[str]) -> List[str]:
    """
    Accept symptom identifiers that are already valid prefixed names in your TTL:
      - wd:Q...
      - sym:...
    Reject plain labels like "fever" because Query 1 can't resolve them without mapping.
    """
    out = []
    for s in symptoms:
        s = s.strip()
        if not s:
            continue
        if s.startswith("wd:Q") or s.startswith("sym:"):
            out.append(s)
        else:
            raise ValueError(
                f"Symptom '{s}' is not a valid prefixed id. "
                f"Pass wd:Q... or sym:... (e.g. ['wd:Q38933', 'sym:swollen_ankles'])."
            )
    return out



def main() -> None:
    # Match your reference file default path logic:
    base_dir = Path(__file__).parent.parent
    rdf_path = base_dir / "ontology" / "databaseV7.ttl"

    g = load_graph(str(rdf_path))

    # --- SAME LOOK AS YOUR REFERENCE OUTPUT ---
    # Reference uses labels: ["fever", "headache"]
    # Query version MUST use wd:Q... ids unless you add mapping
    # Example IDs (you used these earlier):
    symptoms = ["sym:swollen_ankles"]
 # fever, headache
    print(f"\nInput symptoms: {symptoms}\n")

    wd_symps = _symptom_ids_or_passthrough(symptoms)

    # Query 1 gives top diseases + score
    rows = scriptV3.query1_topk_diseases_by_score(
        g=g,
        symps_list=wd_symps,
        top_k_results=5,
        exclude_label=None,
    )

    if not rows:
        return
    

    for i, r in enumerate(rows, 1):
        disease_name = r["label"]
        disease_iri = r["disease_uri"]

        # "Similarity" field in output: use SPARQL finalScore as a percentage
        # (This is not the same as Jaccard, but keeps identical printing format.)
        score = r["finalScore"]
        similarity_pct = (float(score) * 100.0) if score is not None else 0.0

        # For "Matched:" line, call Query 3a to retrieve matching symptom labels
        matched_labels = scriptV3.query3_matching_symptoms(
            g=g,
            disease_iri=disease_iri,
            symps_list=wd_symps,
        )

        print(f"{i}. {disease_name}")
        print(f"   Disease URI: {disease_iri}")
        print(f"   Similarity: {similarity_pct:.2f}%")
        print(f"   Matched: {', '.join(matched_labels)}\n")


if __name__ == "__main__":
    main()
