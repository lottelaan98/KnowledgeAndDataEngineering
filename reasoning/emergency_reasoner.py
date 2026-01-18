import re
from typing import Dict, Tuple, List

import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Tuple
from collections import defaultdict

from rdflib import Graph, Namespace, RDF, OWL, URIRef, Literal
from rdflib.namespace import SKOS, RDFS, XSD

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def classify_see_doctor_text_cosine(
    see_doctor_text: str,
    threshold_unclear: float = 0.20,
) -> Dict[str, float]:
    """
    Returns similarity scores per label:
      {"emergency":0.81, "urgent":0.55, "routine":0.12, "unclear":0.03}

    Pick the max label, but if max < threshold_unclear => "unclear".
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    text = _norm(see_doctor_text)

    # Prototypes: keep these short but expressive
    prototypes: Dict[str, str] = {
        "emergency": _norm(
            "call emergency services immediately now severe sudden life threatening "
            "go to emergency department immediately urgent emergency"
        ),
        "urgent": _norm(
            "contact your doctor urgently today as soon as possible same day "
            "seek medical care soon if worsening"
        ),
        "routine": _norm(
            "make a routine appointment monitor symptoms if persists consult your gp "
            "see doctor if it does not improve"
        ),
        "unclear": _norm(
            "depends unclear not specified consider asking doctor"
        ),
    }

    corpus = [text] + list(prototypes.values())
    vec = TfidfVectorizer(ngram_range=(1,2))
    X = vec.fit_transform(corpus)

    sims = cosine_similarity(X[0], X[1:]).flatten()
    labels = list(prototypes.keys())
    scores = {lab: float(score) for lab, score in zip(labels, sims)}

    # Optionally enforce "unclear" if weak signal
    best_label, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score < threshold_unclear:
        scores["__pred__"] = "unclear"
    else:
        scores["__pred__"] = best_label
    scores["__best_score__"] = best_score
    return scores

seeDoctorText = "Immediately if typhoid is suspected."

scores = classify_see_doctor_text_cosine(seeDoctorText)
pred = scores["__pred__"]
print(pred, scores)

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Set
from rdflib import Graph, Namespace, RDF, URIRef, Literal
from rdflib.namespace import SKOS, RDFS, XSD
import operator
import re

EX = Namespace("http://example.org/med#")

_COMPARATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
}

def _lit_to_float(v: Optional[Literal]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v.toPython())
    except Exception:
        try:
            return float(str(v))
        except Exception:
            return None

def _get_en_label(g: Graph, uri: URIRef) -> str:
    for p in (SKOS.prefLabel, RDFS.label):
        for o in g.objects(uri, p):
            if not getattr(o, "language", None) or o.language == "en":
                return str(o)
    return str(uri)

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@dataclass
class AlarmSymptomDef:
    uri: URIRef
    label: str
    hard_label_uri: Optional[URIRef]
    threshold: Optional[float]
    comparator: Optional[str]
    unit: Optional[str]

@dataclass
class TriageRuleDef:
    uri: URIRef
    label_uri: URIRef
    threshold: int

class KGTriageEngine:
    """
    1) Hard triggers (categorical + numeric thresholds) => immediate label
    2) Else sum scorePoints for matched symptoms => compare to scoreThreshold => urgent/routine
    3) Else unclear
    """

    def __init__(self, graph: Graph):
        self.g = graph
        self.EX = EX

        self.alarm_defs = self._load_alarm_symptoms()
        self.rules = self._load_triage_rules()

        # cache scorePoints for all symptoms
        self.score_points = self._load_score_points()

    def _load_alarm_symptoms(self) -> List[AlarmSymptomDef]:
        defs: List[AlarmSymptomDef] = []
        for s in self.g.subjects(RDF.type, self.EX.AlarmSymptom):
            label = _get_en_label(self.g, s)
            hard_label = next(self.g.objects(s, self.EX.hardTriggerLabel), None)

            threshold = _lit_to_float(next(self.g.objects(s, self.EX.threshold), None))
            comp = next(self.g.objects(s, self.EX.thresholdComparator), None)
            unit = next(self.g.objects(s, self.EX.thresholdUnit), None)

            defs.append(
                AlarmSymptomDef(
                    uri=s,
                    label=label,
                    hard_label_uri=hard_label if isinstance(hard_label, URIRef) else None,
                    threshold=threshold,
                    comparator=str(comp) if comp is not None else None,
                    unit=str(unit) if unit is not None else None,
                )
            )
        return defs

    def _load_triage_rules(self) -> List[TriageRuleDef]:
        out: List[TriageRuleDef] = []
        for r in self.g.subjects(RDF.type, self.EX.TriageRule):
            label_uri = next(self.g.objects(r, self.EX.triageLabel), None)
            thr_lit = next(self.g.objects(r, self.EX.scoreThreshold), None)
            if isinstance(label_uri, URIRef) and thr_lit is not None:
                try:
                    thr = int(thr_lit.toPython())
                except Exception:
                    thr = int(str(thr_lit))
                out.append(TriageRuleDef(uri=r, label_uri=label_uri, threshold=thr))
        return out

    def _load_score_points(self) -> Dict[URIRef, int]:
        pts: Dict[URIRef, int] = {}
        for s, o in self.g.subject_objects(self.EX.scorePoints):
            if isinstance(s, URIRef):
                try:
                    pts[s] = int(o.toPython())
                except Exception:
                    pts[s] = int(str(o))
        return pts

    # -------- matching reported symptoms to alarm symptom defs --------
    def _match_alarm_categorical(
        self, reported_symptom_uris: Set[URIRef]
    ) -> Optional[Tuple[str, URIRef]]:
        """
        If the user reported an AlarmSymptom directly, or a symptom that is closeMatch to an AlarmSymptom,
        trigger its hardTriggerLabel.
        """
        for a in self.alarm_defs:
            if a.hard_label_uri is None:
                continue

            # direct report: user symptom == alarm symptom resource
            if a.uri in reported_symptom_uris:
                return (_get_en_label(self.g, a.hard_label_uri), a.uri)

            # closeMatch bridging: alarmSymptom skos:closeMatch wd:...
            for cm in self.g.objects(a.uri, SKOS.closeMatch):
                if isinstance(cm, URIRef) and cm in reported_symptom_uris:
                    return (_get_en_label(self.g, a.hard_label_uri), a.uri)

        return None

    def _match_alarm_numeric(
        self, measurements: Dict[str, float]
    ) -> Optional[Tuple[str, URIRef, str]]:
        """
        Checks numeric alarm symptoms that have threshold+comparator.
        Map alarm units to measurements keys (you can adjust this mapping).
        """
        # Map ontology unit -> measurement key in your pipeline
        unit_to_key = {
            "C": "temperatureC",
            "0-10": "painScale",
            "mmHg_systolic": "systolicBP",
        }

        for a in self.alarm_defs:
            if a.hard_label_uri is None or a.threshold is None or not a.comparator:
                continue

            key = unit_to_key.get(a.unit or "")
            if not key:
                continue
            if key not in measurements:
                continue

            comp_fn = _COMPARATORS.get(a.comparator)
            if not comp_fn:
                continue

            val = float(measurements[key])
            if comp_fn(val, float(a.threshold)):
                return (_get_en_label(self.g, a.hard_label_uri), a.uri, f"{key}={val} {a.comparator} {a.threshold}")

        return None

    # -------- score points --------
    def score_from_symptoms(self, reported_symptom_uris: Set[URIRef]) -> int:
        score = 0
        for s in reported_symptom_uris:
            score += int(self.score_points.get(s, 0))
        return score

    def triage(
        self,
        reported_symptom_uris: Set[URIRef],
        measurements: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        measurements = measurements or {}

        # 1) numeric hard triggers
        num_hit = self._match_alarm_numeric(measurements)
        if num_hit:
            label, alarm_uri, reason = num_hit
            return {"triage": label, "reason": f"numeric hard trigger: {_get_en_label(self.g, alarm_uri)} ({reason})", "score": None}

        # 2) categorical hard triggers
        cat_hit = self._match_alarm_categorical(reported_symptom_uris)
        if cat_hit:
            label, alarm_uri = cat_hit
            return {"triage": label, "reason": f"categorical hard trigger: {_get_en_label(self.g, alarm_uri)}", "score": None}

        # 3) points-based rule
        total_score = self.score_from_symptoms(reported_symptom_uris)

        # If you only have one rule (adultTriageRule), this picks the first.
        # If you have multiple, you can choose by age group etc.
        if self.rules:
            rule = self.rules[0]
            triage_label = _get_en_label(self.g, rule.label_uri)
            if total_score >= rule.threshold:
                return {"triage": triage_label, "reason": f"score {total_score} >= threshold {rule.threshold}", "score": total_score}
            else:
                return {"triage": "routine", "reason": f"score {total_score} < threshold {rule.threshold}", "score": total_score}

        # 4) fallback
        return {"triage": "unclear", "reason": "no rules / no matches", "score": total_score}

def decide_triage(
    kg_result: Dict[str, Any],
    see_doctor_text: str,
) -> Dict[str, Any]:
    text_scores = classify_see_doctor_text_cosine(see_doctor_text)
    text_pred = text_scores["__pred__"]

    # if KG already says emergency => do not override
    if kg_result["triage"] == "emergency":
        return {
            "triage": "emergency",
            "source": "kg_hard_trigger",
            "kg": kg_result,
            "text_pred": text_pred,
            "text_scores": text_scores,
        }

    # if KG gave urgent/routine with a numeric score, trust KG
    if kg_result["triage"] in {"urgent", "routine"}:
        return {
            "triage": kg_result["triage"],
            "source": "kg_points",
            "kg": kg_result,
            "text_pred": text_pred,
            "text_scores": text_scores,
        }

    # else fallback to text
    return {
        "triage": text_pred,
        "source": "seeDoctor_text_cosine",
        "kg": kg_result,
        "text_pred": text_pred,
        "text_scores": text_scores,
    }

def spacy_lemmas(text: str, nlp=None) -> str:
    if nlp is None:
        return _norm(text)
    doc = nlp(text)
    toks = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
    return " ".join(toks)

q = f"""
PREFIX ex:   <http://example.org/med#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?sym ?label ?points ?hardLabel ?thr ?comp ?unit
WHERE {
  OPTIONAL {
    ?sym ex:scorePoints ?points .
    OPTIONAL { ?sym skos:prefLabel ?label . FILTER(lang(?label)="en") }
  }
  OPTIONAL {
    ?sym a ex:AlarmSymptom .
    OPTIONAL { ?sym skos:prefLabel ?label . FILTER(lang(?label)="en") }
    OPTIONAL { ?sym ex:hardTriggerLabel ?hardLabel . }
    OPTIONAL { ?sym ex:threshold ?thr . }
    OPTIONAL { ?sym ex:thresholdComparator ?comp . }
    OPTIONAL { ?sym ex:thresholdUnit ?unit . }
  }
}
"""