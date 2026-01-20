"""
Triage engine using SPARQL QUERIES (rdflib Graph.query) for:
- Hard trigger alarms (categorical + numeric)
- Sum scorePoints for matched symptoms
- Read triage rule threshold + label (e.g., urgent)
- Read disease ex:seeDoctor text
Then:
- Cosine similarity on triage labels (emergency/urgent/routine/unclear)
- Combine KG + text into one final triage label

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, SKOS


EX = Namespace("http://example.org/med#")


# ----------------------------
# Helpers
# ----------------------------

def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _values_iris(iris: Sequence[str]) -> str:
    # Create "VALUES ?sym { <...> <...> }"
    cleaned = []
    for x in iris:
        if not x:
            continue
        if x.startswith("<") and x.endswith(">"):
            cleaned.append(x)
        else:
            cleaned.append(f"<{x}>")
    return " ".join(cleaned)


def _get_label_text(g: Graph, iri: str) -> str:
    u = URIRef(iri)
    for p in (SKOS.prefLabel, RDFS.label):
        for o in g.objects(u, p):
            if not getattr(o, "language", None) or o.language == "en":
                return str(o)
    return iri


# ----------------------------
# Text cosine triage
# ----------------------------

def triage_by_cosine_text(text: str, threshold_unclear: float = 0.01) -> Dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    text_n = _norm_text(text)

    prototypes = {
        "emergency": _norm_text(
            "call emergency services immediately now life threatening severe sudden "
            "go to emergency department immediately contact emergency"
        ),
        "urgent": _norm_text(
            "contact your doctor urgently today as soon as possible same day "
            "seek medical care soon if worsening"
        ),
        "routine": _norm_text(
            "make a routine appointment monitor symptoms if persists consult your gp "
            "see doctor if it does not improve"
        ),
        "unclear": _norm_text("depends unclear not specified consider asking doctor"),
    }

    corpus = [text_n] + list(prototypes.values())
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(corpus)

    sims = cosine_similarity(X[0], X[1:]).flatten()
    labs = list(prototypes.keys())
    scores = {lab: float(sc) for lab, sc in zip(labs, sims)}

    best_label, best_score = max(scores.items(), key=lambda kv: kv[1])
    # Never "unclear" as basis for cosine. Only emergency/urgent/routine.
    if best_label == "unclear":
        # If "unclear" prototype , degrade to routine
        best_label = "routine"
    return {"pred": best_label, "best_score": best_score, "scores": scores}



# ----------------------------
# KG triage via QUERIES
# ----------------------------

@dataclass
class KGTriageResult:
    pred: str
    score_sum: int
    score_threshold: Optional[int]
    hard_trigger: Optional[str]
    hard_trigger_hits: List[str]
    matched_point_symptoms: List[Tuple[str, int]]  # (label, points)


def query_hard_triggers_categorical(g: Graph, symptom_iris: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Returns list of (alarmSymptomIRI, hardLabelIRI) if:
    - user symptom equals alarm symptom
    OR
    - user symptom equals skos:closeMatch target of alarm symptom
    """
    vals = _values_iris(symptom_iris)
    q = f"""
    PREFIX ex:   <http://example.org/med#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT DISTINCT ?alarm ?triageLabel
    WHERE {{
      VALUES ?sym {{ {vals} }}

      ?alarm rdf:type ex:AlarmSymptom ;
             ex:hardTriggerLabel ?triageLabel .

      FILTER(
        ?sym = ?alarm
        || EXISTS {{ ?alarm skos:closeMatch ?sym }}
      )
    }}
    """
    out = []
    for row in g.query(q):
        out.append((str(row[0]), str(row[1])))
    return out


def query_hard_triggers_numeric(
    g: Graph,
    temperatureC: Optional[float],
    systolicBP: Optional[float],
    painScale: Optional[float],
) -> List[Tuple[str, str]]:
    """
    Checks numeric AlarmSymptom thresholds inside SPARQL.
    Returns list of (alarmIRI, triageLabelIRI) that triggered.
    """
    # If all are missing, skip query
    if temperatureC is None and systolicBP is None and painScale is None:
        return []

    clauses = []
    if temperatureC is not None:
        clauses.append(f'''
        {{
            ?alarm ex:thresholdUnit "C" ;
                ex:threshold ?thr ;
                ex:thresholdComparator ?cmp .
            BIND("{float(temperatureC)}"^^xsd:decimal AS ?val)
        }}
        ''')
    if systolicBP is not None:
        clauses.append(f'''
        {{
            ?alarm ex:thresholdUnit "mmHg_systolic" ;
                ex:threshold ?thr ;
                ex:thresholdComparator ?cmp .
            BIND("{float(systolicBP)}"^^xsd:decimal AS ?val)
        }}
        ''')
    if painScale is not None:
        clauses.append(f'''
        {{
            ?alarm ex:thresholdUnit "0-10" ;
                ex:threshold ?thr ;
                ex:thresholdComparator ?cmp .
            BIND("{float(painScale)}"^^xsd:decimal AS ?val)
        }}
        ''')

    q = f"""
    PREFIX ex:   <http://example.org/med#>
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?alarm ?triageLabel
    WHERE {{
    ?alarm rdf:type ex:AlarmSymptom ;
            ex:hardTriggerLabel ?triageLabel .

    {" UNION ".join(clauses)}

    FILTER(
        (?cmp = ">=" && ?val >= xsd:decimal(?thr)) ||
        (?cmp = ">"  && ?val >  xsd:decimal(?thr)) ||
        (?cmp = "<=" && ?val <= xsd:decimal(?thr)) ||
        (?cmp = "<"  && ?val <  xsd:decimal(?thr)) ||
        (?cmp = "==" && ?val =  xsd:decimal(?thr))
    )
    }}
    """

    out = []
    for row in g.query(q):
        out.append((str(row[0]), str(row[1])))
    return out


def query_score_points_sum(g: Graph, symptom_iris: Sequence[str]) -> Tuple[int, List[Tuple[str, int]]]:
    """
    Sums ex:scorePoints for the matched symptom IRIs.
    Returns (sumPoints, [(symIRI, points), ...])
    """
    vals = _values_iris(symptom_iris)
    q = f"""
    PREFIX ex:   <http://example.org/med#>
    PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

    SELECT ?sym (xsd:integer(?p) AS ?points)
    WHERE {{
      VALUES ?sym {{ {vals} }}
      ?sym ex:scorePoints ?p .
    }}
    """
    hits: List[Tuple[str, int]] = []
    total = 0
    for row in g.query(q):
        sym = str(row[0])
        pts = int(row[1].toPython())
        hits.append((sym, pts))
        total += pts
    hits.sort(key=lambda x: -x[1])
    return total, hits


def query_triage_rule(g: Graph) -> Tuple[Optional[int], Optional[str]]:
    """
    Returns (thresholdInt, triageLabelIRI) from the first ex:TriageRule.
    """
    q = """
    PREFIX ex:   <http://example.org/med#>
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

    SELECT (xsd:integer(?thr) AS ?threshold) ?label
    WHERE {
      ?rule rdf:type ex:TriageRule ;
            ex:scoreThreshold ?thr ;
            ex:triageLabel ?label .
    }
    LIMIT 1
    """
    for row in g.query(q):
        thr = int(row[0].toPython())
        lab = str(row[1])
        return thr, lab
    return None, None


def query_disease_see_doctor(g: Graph, disease_iri: str) -> Optional[str]:
    q = f"""
    PREFIX ex:   <http://example.org/med#>
    SELECT ?txt
    WHERE {{
      <{disease_iri}> ex:seeDoctor ?txt .
    }}
    LIMIT 1
    """
    for row in g.query(q):
        return str(row[0])
    return None


def kg_triage_queries(
    g: Graph,
    symptom_iris: Sequence[str],
    temperatureC: Optional[float] = None,
    systolicBP: Optional[float] = None,
    painScale: Optional[float] = None,
) -> KGTriageResult:
    # 1) Hard triggers (numeric + categorical)
    hard_hits = []
    for alarm_iri, label_iri in query_hard_triggers_numeric(g, temperatureC, systolicBP, painScale):
        hard_hits.append((alarm_iri, label_iri))
    for alarm_iri, label_iri in query_hard_triggers_categorical(g, symptom_iris):
        hard_hits.append((alarm_iri, label_iri))

    # 2) Always sum scorePoints (even if hard trigger fired)
    score_sum, point_hits = query_score_points_sum(g, symptom_iris)
    point_hits_labeled = [(_get_label_text(g, s), p) for s, p in point_hits]

    # If hard trigger fired, return emergency but keep points for explainability
    if hard_hits:
        pred = _get_label_text(g, hard_hits[0][1]).lower()
        alarm_labels = sorted({_get_label_text(g, a) for a, _ in hard_hits})

        return KGTriageResult(
            pred=pred if pred in {"emergency", "urgent", "routine", "unclear"} else "emergency",
            score_sum=score_sum,
            score_threshold=None,          # (optional) you can still read the threshold if you want
            hard_trigger=pred,
            hard_trigger_hits=alarm_labels,
            matched_point_symptoms=point_hits_labeled,
        )

    # 3) Apply triage rule (threshold -> urgent else routine/unclear)
    thr, triage_label_iri = query_triage_rule(g)
    triage_label_txt = _get_label_text(g, triage_label_iri).lower() if triage_label_iri else None

    if thr is not None and triage_label_txt:
        if score_sum >= thr:
            pred = triage_label_txt  # e.g. urgent
        else:
            pred = "routine" if score_sum > 0 else "unclear"

        return KGTriageResult(
            pred=pred,
            score_sum=score_sum,
            score_threshold=thr,
            hard_trigger=None,
            hard_trigger_hits=[],
            matched_point_symptoms=point_hits_labeled,
        )

    # no rule
    return KGTriageResult(
        pred="routine" if score_sum > 0 else "unclear",
        score_sum=score_sum,
        score_threshold=None,
        hard_trigger=None,
        hard_trigger_hits=[],
        matched_point_symptoms=point_hits_labeled,
    )



# ----------------------------
# Combine policies
# ----------------------------

def combine_triage(kg: KGTriageResult, text_pred: Dict[str, Any]) -> Dict[str, Any]:
    text_label = text_pred["pred"]
    best_score = float(text_pred.get("best_score", 0.0))

    # Hard trigger wins
    if kg.hard_trigger or kg.pred == "emergency":
        return {"final": "emergency", "source": "kg_hard_trigger", "kg": kg, "text": text_pred}

    # KG urgent is strong
    if kg.pred == "urgent":
        if text_label == "emergency" and best_score >= 0.35:
            return {"final": "emergency", "source": "text_overrides_kg", "kg": kg, "text": text_pred}
        return {"final": "urgent", "source": "kg_points", "kg": kg, "text": text_pred}

    # KG routine
    if kg.pred == "routine":
        if text_label in {"urgent", "emergency"} and best_score >= 0.35:
            return {"final": text_label, "source": "text_overrides_kg", "kg": kg, "text": text_pred}
        return {"final": "routine", "source": "kg_points_or_default", "kg": kg, "text": text_pred}

    # KG unclear -> rely on text
    return {"final": text_label, "source": "text_only", "kg": kg, "text": text_pred}


def triage_case(
    g: Graph,
    user_text: str,
    symptom_iris: Sequence[str],
    disease_iri: Optional[str] = None,
    temperatureC: Optional[float] = None,
    systolicBP: Optional[float] = None,
    painScale: Optional[float] = None,
    include_user_text_with_doctor_text: bool = True,
    has_symptom_matches: bool = True,
) -> Dict[str, Any]:
    kg = kg_triage_queries(g, symptom_iris, temperatureC, systolicBP, painScale)

    if not has_symptom_matches:
        # No symptom matches out of user input => only unclear
        kg = kg_triage_queries(g, symptom_iris, temperatureC, systolicBP, painScale)
        return {
            "final": "unclear",
            "source": "no_symptoms_matched",
            "kg": kg,
            "text": {"pred": "unclear", "best_score": 0.0, "scores": {}},
            "seeDoctorText": None,
            "textUsed": user_text,
        }

    text_source = user_text
    see_doc = None
    if disease_iri:
        see_doc = query_disease_see_doctor(g, disease_iri)
        if see_doc and include_user_text_with_doctor_text:
            text_source = (see_doc + " " + user_text).strip()
        elif see_doc:
            text_source = see_doc

    text_pred = triage_by_cosine_text(text_source)
    out = combine_triage(kg, text_pred)
    out["seeDoctorText"] = see_doc
    out["textUsed"] = text_source
    return out


# ----------------------------
# Demo run (no CLI args)
# ----------------------------

def main():
    base_dir = Path(__file__).resolve().parent.parent
    ttl = base_dir / "ontology" / "databaseV7.ttl"

    g = Graph()
    g.parse(str(ttl), format="turtle")

    # Example: user input -> you normally fill these from your pipeline

    # Example: 1
    # user_text = "I have been experiencing a skin rash. It is red and itchy."
    # symptom_iris = [
    #     "http://www.wikidata.org/entity/Q653197",   # rash
    #     "http://www.wikidata.org/entity/Q199602",   # itch
    # ]

    # disease_iri = "http://www.wikidata.org/entity/Q179945"  # Psoriasis (from your TTL)

    # Example 2
    user_text = "I have chest pain for 10 minutes and feel confused."
    symptom_iris =  ['http://www.wikidata.org/entity/Q693058', 'http://www.wikidata.org/entity/Q35805', 'http://www.wikidata.org/entity/Q38933', 'http://www.wikidata.org/entity/Q81938']
    #[
    #     "http://www.wikidata.org/entity/Q693058",  # chest pain
    #     #"http://www.wikidata.org/entity/Q557945",  # confusion
    # ]

    disease_iri = "http://www.wikidata.org/entity/Q83319"  # e.g., Typhoid (example)

    # Optional numeric fields (None if empty)
    temperatureC = 41
    systolicBP = 150
    painScale = 8

    result = triage_case(
        g=g,
        user_text=user_text,
        symptom_iris=symptom_iris,
        disease_iri=disease_iri,
        temperatureC=temperatureC,
        systolicBP=systolicBP,
        painScale=painScale,
    )

    print("\n=== FINAL TRIAGE ===")
    print("final:", result["final"], "| source:", result["source"])

    print("\n--- KG ---")
    print("kg pred:", result["kg"].pred)
    print("hard triggers:", result["kg"].hard_trigger_hits)
    print("score sum:", result["kg"].score_sum, "threshold:", result["kg"].score_threshold)
    print("point hits:", result["kg"].matched_point_symptoms)

    print("\n--- TEXT ---")
    print("text pred:", result["text"]["pred"], "best:", f"{result['text']['best_score']:.3f}")
    print("scores:", result["text"]["scores"])

    if result.get("seeDoctorText"):
        print("\nseeDoctor text:", result["seeDoctorText"])


if __name__ == "__main__":
    main()