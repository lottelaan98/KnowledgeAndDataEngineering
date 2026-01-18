"""
Triage engine that combines:
A) Knowledge-Graph rule/points triage
B) Text cosine similarity triage (emergency/urgent/routine/unclear)

Inputs you provide from your pipeline:
- user_text: str
- symptom_uris: List[str] or Set[str]  (e.g., "http://www.wikidata.org/entity/Q38933" or "http://example.org/med#symptom/red_skin")
- disease_uri: str (the matched disease URI)
- temperatureC: Optional[float]
- systolicBP: Optional[float]
- painScale: Optional[float]

Outputs:
- final triage label: emergency | urgent | routine | unclear
- debug info (scores, triggers, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _get_en_label(g: Graph, uri: URIRef) -> str:
    for p in (SKOS.prefLabel, RDFS.label):
        for o in g.objects(uri, p):
            if not getattr(o, "language", None) or o.language == "en":
                return str(o)
    return str(uri)


def _to_uriref_list(items: Set[str] | List[str]) -> List[URIRef]:
    out = []
    for x in items:
        if not x:
            continue
        out.append(URIRef(x))
    return out


# ----------------------------
# Text cosine triage
# ----------------------------

def triage_by_cosine_text(
    text: str,
    threshold_unclear: float = 0.20,
) -> Dict[str, Any]:
    """
    Classify text into emergency/urgent/routine/unclear using TF-IDF cosine similarity
    against label prototypes.

    Returns:
      {
        "pred": "urgent",
        "best_score": 0.52,
        "scores": {"emergency":..., "urgent":..., ...}
      }
    """
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
        "unclear": _norm_text(
            "depends unclear not specified consider asking doctor"
        ),
    }

    corpus = [text_n] + list(prototypes.values())
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(corpus)

    sims = cosine_similarity(X[0], X[1:]).flatten()
    labels = list(prototypes.keys())
    scores = {lab: float(sc) for lab, sc in zip(labels, sims)}

    best_label, best_score = max(scores.items(), key=lambda kv: kv[1])
    pred = best_label if best_score >= threshold_unclear else "unclear"

    return {"pred": pred, "best_score": best_score, "scores": scores}


# ----------------------------
# KG triage (hard triggers + points)
# ----------------------------

@dataclass
class KGTriageResult:
    pred: str                      # emergency/urgent/routine/unclear
    score_sum: int                 # sum(scorePoints)
    score_threshold: Optional[int] # threshold from rule, if exists
    hard_trigger: Optional[str]    # emergency if triggered
    hard_trigger_symptoms: List[str]
    matched_point_symptoms: List[Tuple[str, int]]  # (label, points)


def kg_triage(
    g: Graph,
    symptom_uris: Set[str] | List[str],
    temperatureC: Optional[float] = None,
    systolicBP: Optional[float] = None,
    painScale: Optional[float] = None,
) -> KGTriageResult:
    """
    Implements:
    1) Hard trigger alarm symptoms:
       - categorical: symptom is AlarmSymptom (or is skos:closeMatch of one)
       - numeric: alarm symptom has thresholdComparator/unit and you provided measurement
    2) Else sum ex:scorePoints over symptoms
    3) Compare to first ex:TriageRule scoreThreshold and triageLabel (e.g. urgent)
    4) If no rule / no matches -> unclear/routine fallback
    """
    sym_refs = _to_uriref_list(set(symptom_uris))

    # --- 1) Numeric alarm checks ---
    # We map alarm unit -> measurement value
    unit_to_value = {
        "C": temperatureC,
        "mmHg_systolic": systolicBP,
        "0-10": painScale,
    }

    # Find all alarm symptoms
    alarm_symptoms = list(g.subjects(RDF.type, EX.AlarmSymptom))

    hard_hits: List[str] = []

    # numeric triggers
    for alarm in alarm_symptoms:
        hard_label_uri = next(g.objects(alarm, EX.hardTriggerLabel), None)
        if hard_label_uri is None:
            continue

        thr = next(g.objects(alarm, EX.threshold), None)
        comp = next(g.objects(alarm, EX.thresholdComparator), None)
        unit = next(g.objects(alarm, EX.thresholdUnit), None)

        # Only numeric alarms have all of these
        if thr is None or comp is None or unit is None:
            continue

        unit_s = str(unit)
        val = unit_to_value.get(unit_s)
        if val is None:
            continue

        try:
            thr_f = float(thr.toPython())
        except Exception:
            try:
                thr_f = float(str(thr))
            except Exception:
                continue

        comp_s = str(comp).strip()

        def cmp(a: float, b: float) -> bool:
            if comp_s == ">=":
                return a >= b
            if comp_s == ">":
                return a > b
            if comp_s == "<=":
                return a <= b
            if comp_s == "<":
                return a < b
            if comp_s == "==":
                return a == b
            return False

        if cmp(float(val), float(thr_f)):
            hard_hits.append(_get_en_label(g, alarm))

    # categorical triggers: if user symptom is the alarm symptom OR skos:closeMatch of it
    for alarm in alarm_symptoms:
        hard_label_uri = next(g.objects(alarm, EX.hardTriggerLabel), None)
        if hard_label_uri is None:
            continue

        alarm_uri = URIRef(str(alarm))
        close_matches = set(g.objects(alarm, SKOS.closeMatch))

        for s in sym_refs:
            if s == alarm_uri or s in close_matches:
                hard_hits.append(_get_en_label(g, alarm))

    # If any hard hit exists, force emergency (or whatever label is on the alarm)
    # NOTE: your TTL sets hardTriggerLabel ex:emergency for alarms
    if hard_hits:
        return KGTriageResult(
            pred="emergency",
            score_sum=0,
            score_threshold=None,
            hard_trigger="emergency",
            hard_trigger_symptoms=sorted(set(hard_hits)),
            matched_point_symptoms=[],
        )

    # --- 2) Score points sum ---
    score_sum = 0
    point_hits: List[Tuple[str, int]] = []

    for s in sym_refs:
        pts_lit = next(g.objects(s, EX.scorePoints), None)
        if pts_lit is None:
            continue
        try:
            pts = int(pts_lit.toPython())
        except Exception:
            try:
                pts = int(str(pts_lit))
            except Exception:
                continue

        score_sum += pts
        point_hits.append((_get_en_label(g, s), pts))

    # --- 3) Load triage rule (first rule; you can expand later) ---
    rule = next(g.subjects(RDF.type, EX.TriageRule), None)
    if rule is not None:
        thr_lit = next(g.objects(rule, EX.scoreThreshold), None)
        triage_label_uri = next(g.objects(rule, EX.triageLabel), None)

        thr = None
        if thr_lit is not None:
            try:
                thr = int(thr_lit.toPython())
            except Exception:
                thr = int(str(thr_lit))

        label = "urgent"
        if isinstance(triage_label_uri, URIRef):
            label = _get_en_label(g, triage_label_uri).lower()

        if thr is not None and score_sum >= thr:
            return KGTriageResult(
                pred=label,  # typically "urgent"
                score_sum=score_sum,
                score_threshold=thr,
                hard_trigger=None,
                hard_trigger_symptoms=[],
                matched_point_symptoms=sorted(point_hits, key=lambda x: -x[1]),
            )

        # Below threshold -> routine if we had any point evidence, else unclear
        return KGTriageResult(
            pred="routine" if (score_sum > 0) else "unclear",
            score_sum=score_sum,
            score_threshold=thr,
            hard_trigger=None,
            hard_trigger_symptoms=[],
            matched_point_symptoms=sorted(point_hits, key=lambda x: -x[1]),
        )

    # No rules exist -> fallback
    return KGTriageResult(
        pred="routine" if (score_sum > 0) else "unclear",
        score_sum=score_sum,
        score_threshold=None,
        hard_trigger=None,
        hard_trigger_symptoms=[],
        matched_point_symptoms=sorted(point_hits, key=lambda x: -x[1]),
    )


# ----------------------------
# Combine KG + cosine into final triage
# ----------------------------

def combine_triage(
    kg: KGTriageResult,
    text_pred: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combination policy (simple + safe):

    1) If KG hard trigger => emergency always
    2) Else if KG says urgent => urgent unless text strongly says emergency
    3) Else if KG says routine:
         - if text says urgent/emergency with good confidence => take text
         - else routine
    4) Else unclear => take text

    You can adjust the thresholds easily.
    """
    text_label = text_pred["pred"]
    best_score = float(text_pred.get("best_score", 0.0))

    # 1) hard trigger wins
    if kg.hard_trigger == "emergency" or kg.pred == "emergency":
        final_label = "emergency"
        source = "kg_hard_trigger"
        return {"final": final_label, "source": source, "kg": kg, "text": text_pred}

    # 2) KG urgent
    if kg.pred == "urgent":
        if text_label == "emergency" and best_score >= 0.35:
            return {"final": "emergency", "source": "text_overrides_kg", "kg": kg, "text": text_pred}
        return {"final": "urgent", "source": "kg_points", "kg": kg, "text": text_pred}

    # 3) KG routine
    if kg.pred == "routine":
        if text_label in {"urgent", "emergency"} and best_score >= 0.35:
            return {"final": text_label, "source": "text_overrides_kg", "kg": kg, "text": text_pred}
        return {"final": "routine", "source": "kg_points_or_default", "kg": kg, "text": text_pred}

    # 4) KG unclear -> trust text
    return {"final": text_label, "source": "text_only", "kg": kg, "text": text_pred}


# ----------------------------
# Convenience: full triage call
# ----------------------------

def triage_case(
    g: Graph,
    user_text: str,
    symptom_uris: Set[str] | List[str],
    disease_uri: Optional[str] = None,
    temperatureC: Optional[float] = None,
    systolicBP: Optional[float] = None,
    painScale: Optional[float] = None,
    use_disease_see_doctor_text: bool = True,
) -> Dict[str, Any]:
    """
    Main function you call from main.py.

    - If disease_uri is provided and use_disease_see_doctor_text=True:
        we fetch ex:seeDoctor text from KG and classify that text
      Otherwise:
        we classify user_text
    """
    # KG triage
    kg_res = kg_triage(
        g=g,
        symptom_uris=symptom_uris,
        temperatureC=temperatureC,
        systolicBP=systolicBP,
        painScale=painScale,
    )

    # text source
    text_for_triage = user_text
    see_doctor_text = None

    if disease_uri and use_disease_see_doctor_text:
        d = URIRef(disease_uri)
        lit = next(g.objects(d, EX.seeDoctor), None)
        if lit is not None:
            see_doctor_text = str(lit)
            # Combine both helps a bit
            text_for_triage = (see_doctor_text + " " + user_text).strip()

    text_pred = triage_by_cosine_text(text_for_triage)

    combined = combine_triage(kg_res, text_pred)
    combined["seeDoctorText"] = see_doctor_text
    combined["textUsed"] = text_for_triage

    return combined


# ----------------------------
# Demo (runs without any args)
# ----------------------------

def main():
    # Example only; in your main.py you will call triage_case(...)
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent
    ttl = base_dir / "ontology" / "databaseV7.ttl"

    g = Graph()
    g.parse(str(ttl), format="turtle")

    user_text = "I have chest pain for 10 minutes and feel confused."
    symptom_uris = [
        "http://www.wikidata.org/entity/Q693058",  # chest pain
        "http://www.wikidata.org/entity/Q557945",  # confusion
    ]

    disease_uri = "http://www.wikidata.org/entity/Q83319"  # e.g., Typhoid (example)

    out = triage_case(
        g=g,
        user_text=user_text,
        symptom_uris=symptom_uris,
        disease_uri=disease_uri,
        temperatureC=None,
        systolicBP=None,
        painScale=None,
    )

    print("\n=== TRIAGE RESULT ===")
    print("Final:", out["final"], "| source:", out["source"])
    print("\n--- KG ---")
    print("KG pred:", out["kg"].pred)
    print("Hard triggers:", out["kg"].hard_trigger_symptoms)
    print("Score sum:", out["kg"].score_sum, "threshold:", out["kg"].score_threshold)
    print("Point hits:", out["kg"].matched_point_symptoms)
    print("\n--- TEXT ---")
    print("Text pred:", out["text"]["pred"], "best_score:", f"{out['text']['best_score']:.3f}")
    print("Scores:", out["text"]["scores"])
    if out.get("seeDoctorText"):
        print("\nseeDoctor:", out["seeDoctorText"])


if __name__ == "__main__":
    main()
