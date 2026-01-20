"""
scriptV2.py — Query utilities (import-friendly)

This rewrites your notebook-style script into reusable functions:
- load_graph(ttl_path)
- query1_topk_diseases_by_score(g, symps_list, top_k_results=3, exclude_label=None)
- query2_disease_wiki_info(g, disease_iri)
- query3_matching_symptoms(g, disease_iri, symps_list)
- query3_missing_symptoms(g, disease_iri, symps_list)

So your *main* file can do:
    from scriptV2 import load_graph, query1_topk_diseases_by_score, ...
and it won’t contain big SPARQL strings anymore.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from rdflib import Graph
from rdflib.plugins.parsers.notation3 import BadSyntax


# -------------------------
# Load / parse TTL
# -------------------------
def load_graph(ttl_path: str) -> Graph:
    """
    Parse a Turtle file into an rdflib Graph and return it.
    Raises exceptions if parsing fails (caller can catch).
    """
    g = Graph()
    g.parse(ttl_path, format="turtle")
    return g


def safe_load_graph(ttl_path: str) -> Optional[Graph]:
    """
    Convenience wrapper: prints errors and returns None on failure.
    """
    try:
        g = load_graph(ttl_path)
        print("OK. Triples:", len(g))
        return g
    except BadSyntax as e:
        print("BadSyntax:", e)
    except Exception as e:
        print(type(e).__name__, e)
    return None


# -------------------------
# Helpers
# -------------------------
def _symps_to_values(symps_list: List[str]) -> str:
    """
    Convert ["wd:Q38933", "wd:Q9690"] -> "wd:Q38933 wd:Q9690"
    Assumes caller passes valid prefixed names.
    """
    return " ".join(s.strip() for s in symps_list if s and s.strip())


def _iri_to_sparql_iri(disease_iri: str) -> str:
    """
    Accepts either:
      - "http://www.wikidata.org/entity/Q83319"
      - "<http://www.wikidata.org/entity/Q83319>"
    Returns a SPARQL-safe IRI in angle brackets.
    """
    d = disease_iri.strip()
    if d.startswith("<") and d.endswith(">"):
        return d
    return f"<{d.strip('<>')}>"


# ============================================================
# QUERY 1: Top-k diseases by score (+ optional exclude label)
# ============================================================
def query1_topk_diseases_by_score(
    g: Graph,
    symps_list: List[str],
    top_k_results: int = 3,
    exclude_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
      {
        "disease_uri": "...",
        "label": "...",
        "catNorm": float,
        "baseScore": float,
        "finalScore": float
      }
    """

    input_symps = _symps_to_values(symps_list)
    if not input_symps:
        return []

    exclude_filter = ""
    if exclude_label and exclude_label.strip():
        # Case-insensitive exclude on label
        # Note: if exclude_label contains double quotes, you should escape them.
        ex = exclude_label.strip().replace('"', '\\"')
        exclude_filter = f'FILTER(LCASE(STR(?label)) != LCASE("{ex}"))'

    q = f"""
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX ex:   <http://example.org/med#>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX sym:  <http://example.org/med#symptom/>

SELECT ?disease ?label
       ?catNorm
       ?baseScore ?finalScore
WHERE {{

  ### USER INPUT SYMPTOMS ###
  VALUES ?inSym {{ {input_symps} }}

  ### DISEASE + LABEL ###
  ?disease a ex:Disease ;
           skos:prefLabel ?label .
  FILTER(lang(?label) = "en") 
  {exclude_filter}



  ### COUNT INPUT SYMPTOMS ###
  {{
    SELECT (COUNT(DISTINCT ?in) AS ?inputCount)
    WHERE {{
      VALUES ?in {{ {input_symps} }}
    }}
  }}

  ### MATCHED COUNT: COUNT INPUT SYMPTOMS EXISTING IN DEASEASE SYMPTOMATOLOGY ###
  {{
    SELECT ?disease (COUNT(DISTINCT ?m) AS ?matchedCount)
    WHERE {{
      VALUES ?m {{ {input_symps} }}
      ?disease (ex:hasPrimarySymptom|ex:hasSecondarySymptom) ?m .
    }}
    GROUP BY ?disease
  }}

  ### DISEASE TOTAL SYMPTOMS ###
  {{
    SELECT ?disease (COUNT(DISTINCT ?sAll) AS ?diseaseSymptomCount)
    WHERE {{
      ?disease (ex:hasPrimarySymptom|ex:hasSecondarySymptom) ?sAll .
    }}
    GROUP BY ?disease
  }}

  ### JACCARD SIMILARITY ###
  BIND(
    IF(
      (xsd:decimal(?inputCount) + xsd:decimal(?diseaseSymptomCount) - xsd:decimal(?matchedCount)) = 0,
      0.0,
      xsd:decimal(?matchedCount) /
      (xsd:decimal(?inputCount) + xsd:decimal(?diseaseSymptomCount) - xsd:decimal(?matchedCount))
    ) AS ?jaccard
  )

  ### COVERAGE ###
  BIND(
    IF(
      xsd:decimal(?diseaseSymptomCount) = 0,
      0.0,
      xsd:decimal(?matchedCount) / xsd:decimal(?diseaseSymptomCount)
    ) AS ?coverage
  )


  ### RARE SYMPTOM CONTRIBUTION PER DISEASE ###

  ### IDF RAW: idfRaw(disease) = 1/df(m), df(m) = #diseases containing symptom m ###

  {{
    SELECT ?disease (SUM(?w) AS ?idfRaw)
    WHERE {{
      VALUES ?t {{ {input_symps} }}
      ?disease (ex:hasPrimarySymptom|ex:hasSecondarySymptom) ?t .

      {{
        SELECT ?t (COUNT(DISTINCT ?d2) AS ?df)
        WHERE {{
          ?d2 a ex:Disease ;
              (ex:hasPrimarySymptom|ex:hasSecondarySymptom) ?t .
        }}
        GROUP BY ?t
      }}

      BIND( IF(?df = 0, 0.0, (1.0 / xsd:decimal(?df))) AS ?w )
    }}
    GROUP BY ?disease
  }}

  ### IDF MAX: idfMax = SUM_ ( x in input) --- (1/df(x))  (upper bound for this input) ###
  {{
    SELECT (SUM(?wIn) AS ?idfMax)
    WHERE {{
      VALUES ?x {{ {input_symps} }}

      {{
        SELECT ?x (COUNT(DISTINCT ?d3) AS ?dfIn)
        WHERE {{
          ?d3 a ex:Disease ;
              (ex:hasPrimarySymptom|ex:hasSecondarySymptom) ?x .
        }}
        GROUP BY ?x
      }}

      BIND( IF(?dfIn = 0, 0.0, (1.0 / xsd:decimal(?dfIn))) AS ?wIn )
    }}
  }}

  BIND( IF(?idfMax = 0, 0.0, xsd:decimal(?idfRaw) / xsd:decimal(?idfMax)) AS ?idfNorm )


  ### PRIMARY SYMPTOMS MENTIONED IN USER INPUT ###
  OPTIONAL {{
    SELECT ?disease (COUNT(DISTINCT ?cm) AS ?coreMatchedRaw)
    WHERE {{
      VALUES ?cm {{ {input_symps} }}
      ?disease ex:hasPrimarySymptom ?cm .
    }}
    GROUP BY ?disease
  }}
  BIND(COALESCE(?coreMatchedRaw, 0) AS ?coreMatched)

  BIND(
    IF(?coreMatched >= 3, 1.00,
      IF(?coreMatched = 2, 0.90,
        IF(?coreMatched = 1, 0.70, 0.40)
      )
    ) AS ?coreCoeff
  )

  ### SYMPTOM TYPE BONUS ###
  {{
    SELECT (COUNT(DISTINCT ?pType) AS ?patientCatCount)
    WHERE {{
      VALUES ?s {{ {input_symps} }}
      ?s a ?pType .
      ?pType rdfs:subClassOf ex:Symptom .
    }}
  }}

  OPTIONAL {{
    SELECT ?disease (COUNT(DISTINCT ?overCat) AS ?catOverlapRaw)
    WHERE {{
      VALUES ?s {{ {input_symps} }}

      ?s a ?pType .
      ?pType rdfs:subClassOf ex:Symptom .

      ?disease (ex:hasPrimarySymptom|ex:hasSecondarySymptom) ?ds .
      ?ds a ?dType .
      ?dType rdfs:subClassOf ex:Symptom .

      FILTER(?pType = ?dType)
      BIND(?pType AS ?overCat)
    }}
    GROUP BY ?disease
  }}
  BIND(COALESCE(?catOverlapRaw, 0) AS ?catOverlap)

  BIND(
    IF(?patientCatCount = 0, 0.0,
      xsd:decimal(?catOverlap) / xsd:decimal(?patientCatCount)
    ) AS ?catNorm
  )

  ### TOTAL SCORE FORMULA ###
  BIND( (0.6 * ?idfNorm + 0.2 * ?jaccard + 0.2 * ?coverage) AS ?baseScore )
  BIND( (?baseScore * ?coreCoeff + 0.10 * ?catNorm) AS ?finalScore )
}}
GROUP BY ?disease

ORDER BY DESC(?finalScore)
LIMIT {top_k_results}
"""

    out: List[Dict[str, Any]] = []
    for row in g.query(q):
        # row: (?disease ?label ?catNorm ?baseScore ?finalScore)
        out.append(
            {
                "disease_uri": str(row[0]),
                "label": str(row[1]),
                "catNorm": float(row[2]) if row[2] is not None else None,
                "baseScore": float(row[3]) if row[3] is not None else None,
                "finalScore": float(row[4]) if row[4] is not None else None,
            }
        )
    return out


# ==========================================
# QUERY 2: Disease wiki/info by disease IRI
# ==========================================
def query2_disease_wiki_info(g: Graph, disease_iri: str) -> Optional[Dict[str, str]]:
    disease = _iri_to_sparql_iri(disease_iri)

    q = f"""
PREFIX ex:  <http://example.org/med#>
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?label ?explanation ?visitDoctor ?treatment
WHERE {{
      {disease} a ex:Disease ;
                skos:prefLabel ?label ;
                ex:hasExplanation ?explanation ;
                ex:seeDoctor ?visitDoctor ;
                ex:treatment ?treatment .
}}
"""
    for row in g.query(q):
        return {
            "label": str(row[0]),
            "explanation": str(row[1]),
            "seeDoctor": str(row[2]),
            "treatment": str(row[3]),
        }
    return None


# ==========================================
# QUERY 3a: Matching symptoms
# ==========================================
def query3_matching_symptoms(g: Graph, disease_iri: str, symps_list: List[str]) -> List[str]:
    disease = _iri_to_sparql_iri(disease_iri)
    input_symps = _symps_to_values(symps_list)
    if not input_symps:
        return []

    q = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX ex:   <http://example.org/med#>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX sym:  <http://example.org/med#symptom/>

SELECT ?SympName
WHERE {{
  VALUES ?inSym {{ {input_symps} }}

  {disease} a ex:Disease ;
            skos:prefLabel ?label .
  FILTER(lang(?label) = "en")

  {disease} (ex:hasPrimarySymptom|ex:hasSecondarySymptom) ?inSym .
  ?inSym skos:prefLabel ?SympName .
  FILTER(lang(?SympName) = "en")
}}
GROUP BY ?SympName
ORDER BY ?SympName
LIMIT 200
"""
    return [str(row[0]) for row in g.query(q)]


# ==========================================
# QUERY 3b: Missing symptoms (disease - user)
# ==========================================
def query3_missing_symptoms(g: Graph, disease_iri: str, symps_list: List[str]) -> List[str]:
    disease = _iri_to_sparql_iri(disease_iri)
    input_symps = _symps_to_values(symps_list)
    if not input_symps:
        return []

    q = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX ex:   <http://example.org/med#>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX sym:  <http://example.org/med#symptom/>

SELECT ?SympName
WHERE {{
  {disease} a ex:Disease ;
            skos:prefLabel ?label .
  FILTER(lang(?label) = "en")

  {disease} (ex:hasPrimarySymptom|ex:hasSecondarySymptom) ?symptom .
  ?symptom skos:prefLabel ?SympName .
  FILTER(lang(?SympName) = "en")

  MINUS {{
    VALUES ?symptom {{ {input_symps} }}
  }}
}}
GROUP BY ?SympName
ORDER BY ?SympName
LIMIT 200
"""
    return [str(row[0]) for row in g.query(q)]


if __name__ == "__main__":
    from pathlib import Path

    # ------------------------------------------------------------
    # Resolve project root automatically
    # (expects ontology/databaseV7.ttl to exist)
    # ------------------------------------------------------------
    here = Path(__file__).resolve()
    project_root = here.parent

    # Walk up until we find ontology/databaseV7.ttl
    for _ in range(10):
        if (project_root / "ontology" / "databaseV7.ttl").exists():
            break
        if project_root.parent == project_root:
            print("Could not locate ontology/databaseV7.ttl")
            raise SystemExit(1)
        project_root = project_root.parent

    ttl = project_root / "ontology" / "databaseV7.ttl"

    # ------------------------------------------------------------
    # Load graph safely
    # ------------------------------------------------------------
    g = safe_load_graph(str(ttl))
    if g is None:
        raise SystemExit(1)

    # ------------------------------------------------------------
    # DEMO: Query 1
    # ------------------------------------------------------------
    print("\n--- DEMO: Query 1 ---")
    symps_q1 = ["sym:swollen_ankles", "wd:Q35805", 'wd:Q38933', 'wd:Q81938']
    exc = "Typhoid"

    rows = query1_topk_diseases_by_score(
        g,
        symps_q1,
        top_k_results=3,
        exclude_label=exc,
    )

    for i, r in enumerate(rows, 1):
        print(f"{i}. Disease: {r['label']}   Score: {r['finalScore']:.3f}")
        print(f"   URI: {r['disease_uri']}")

    # ------------------------------------------------------------
    # DEMO: Query 2
    # ------------------------------------------------------------
    disease_iri = "http://www.wikidata.org/entity/Q83319"

    print("\n--- DEMO: Query 2 ---")
    info = query2_disease_wiki_info(g, disease_iri)
    if info:
        print(
            f"Disease: {info['label']}\n"
            f"Explanation: {info['explanation']}\n"
            f"Visit to the doctor: {info['seeDoctor']}\n"
            f"Treatment: {info['treatment']}"
        )
    else:
        print("(no info found)")

    # ------------------------------------------------------------
    # DEMO: Query 3
    # ------------------------------------------------------------
    print("\n--- DEMO: Query 3 ---")
    symps_q3 = [
        "wd:Q2536390",
        "wd:Q29644032",
        "wd:Q38933",
        "wd:Q9690",
        "wd:Q86",
        "wd:Q87",
    ]

    matched = query3_matching_symptoms(g, disease_iri, symps_q3)
    missing = query3_missing_symptoms(g, disease_iri, symps_q3)

    print("Matched symptoms:")
    for s in matched:
        print("-", s)

    print("\nMissing symptoms:")
    for s in missing:
        print("-", s)

