"""
RDF-based disease finder aligned with the refactored medical ontology.
Supports symptom-role aware querying and similarity scoring (PURE PYTHON),
AND includes SPARQL query utilities (QUERY 1/2/3) like your Colab script.

What you get in this one file:
1) RDFDiseaseFinder.find_nearest_diseases(...)   # Python scoring (role-weighted + multiword bonus + normalized)
2) RDFDiseaseFinder.query_topk_diseases_by_score(...)  # SPARQL Query 1
3) RDFDiseaseFinder.query_disease_wiki_info(...)       # SPARQL Query 2
4) RDFDiseaseFinder.query_matching_symptoms(...)       # SPARQL Query 3a
5) RDFDiseaseFinder.query_missing_symptoms(...)        # SPARQL Query 3b

Notes:
- Works with rdflib Graph() already loaded from your TTL.
- The SPARQL queries expect input symptoms as URIs like "wd:Q38933" etc.
- Excluding a disease by label is supported in Query 1 (case-insensitive).
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Tuple
from collections import defaultdict

from rdflib import Graph, Namespace, RDF, OWL, URIRef, Literal
from rdflib.namespace import SKOS, RDFS, XSD


class RDFDiseaseFinder:
    # ---------------------------
    # Role weights (Python scorer)
    # ---------------------------
    WEIGHT_RARE = 3
    WEIGHT_PRIMARY = 2
    WEIGHT_SECONDARY = 1

    # Multi-word bonus:
    #  - 1 word: 1.00x
    #  - 2 words: 1.25x
    #  - 3 words: 1.50x
    #  - 4 words: 1.75x
    WORD_BONUS_ALPHA = 0.25

    def __init__(self, rdf_path: str):
        self.graph = Graph()
        self.graph.parse(rdf_path, format="turtle")

        # Namespaces
        self.EX = Namespace("http://example.org/med#")
        self.WD = Namespace("http://www.wikidata.org/entity/")

        self.graph.bind("ex", self.EX)
        self.graph.bind("wd", self.WD)
        self.graph.bind("skos", SKOS)

        # Caches
        self._symptom_label_cache: Dict[str, str] = {}
        self._disease_label_cache: Dict[str, str] = {}

        # Sanity check
        if not any(self.graph.triples((None, RDF.type, self.EX.Disease))):
            raise RuntimeError("No ex:Disease instances found. Check ontology or namespace.")

        # Cache role-aware index
        self._disease_symptoms_cache = self.get_all_disease_symptoms()

    # ------------------------------------------------------------------
    # Normalization helpers (Python scorer)
    # ------------------------------------------------------------------
    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"[-_]+", " ", text.lower().strip())

    @staticmethod
    def _count_words(text: str) -> int:
        t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        t = re.sub(r"\s+", " ", t).strip()
        return len([w for w in t.split(" ") if w])

    def _symptom_word_multiplier(self, symptom_uri: URIRef) -> float:
        label = self.get_symptom_label(symptom_uri)
        n_words = self._count_words(label)
        if n_words <= 1:
            return 1.0
        return 1.0 + self.WORD_BONUS_ALPHA * (n_words - 1)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    def _get_label(self, uri: URIRef, cache: Dict[str, str]) -> str:
        uri_str = str(uri)
        if uri_str in cache:
            return cache[uri_str]

        for p in (SKOS.prefLabel, RDFS.label):
            for label in self.graph.objects(uri, p):
                if not getattr(label, "language", None) or label.language == "en":
                    cache[uri_str] = str(label)
                    return cache[uri_str]

        # fallback: localname
        cache[uri_str] = uri_str.split("/")[-1]
        return cache[uri_str]

    def get_symptom_label(self, uri: URIRef) -> str:
        return self._get_label(uri, self._symptom_label_cache)

    def get_disease_label(self, uri: URIRef) -> str:
        return self._get_label(uri, self._disease_label_cache)

    # ------------------------------------------------------------------
    # Symptom lookup (Python scorer expects symptom labels)
    # ------------------------------------------------------------------
    def find_symptom_uris(self, symptom_labels: List[str]) -> Set[URIRef]:
        normalized_inputs = {self.normalize(s) for s in symptom_labels}
        matches: Set[URIRef] = set()

        symptom_types = {self.EX.Symptom}
        for subclass in self.graph.subjects(RDFS.subClassOf, self.EX.Symptom):
            symptom_types.add(subclass)

        for symptom_type in symptom_types:
            for symptom in self.graph.subjects(RDF.type, symptom_type):
                label = self.normalize(self.get_symptom_label(symptom))
                for s in normalized_inputs:
                    if s == label or s in label or label in s:
                        matches.add(symptom)

        return matches

    # ------------------------------------------------------------------
    # Disease → symptoms (ROLE-AWARE) (Python scorer)
    # ------------------------------------------------------------------
    def get_all_disease_symptoms(self) -> Dict[URIRef, Dict[str, Set[URIRef]]]:
        disease_symptoms: Dict[URIRef, Dict[str, Set[URIRef]]] = defaultdict(
            lambda: {"primary": set(), "secondary": set(), "rare": set()}
        )

        for disease in self.graph.subjects(RDF.type, self.EX.Disease):
            for s in self.graph.objects(disease, self.EX.hasPrimarySymptom):
                disease_symptoms[disease]["primary"].add(s)
            for s in self.graph.objects(disease, self.EX.hasSecondarySymptom):
                disease_symptoms[disease]["secondary"].add(s)
            # If your KG uses hasRareSymptom:
            for s in self.graph.objects(disease, self.EX.hasRareSymptom):
                disease_symptoms[disease]["rare"].add(s)

        return disease_symptoms

    # ------------------------------------------------------------------
    # Similarity search (PYTHON) (ROLE-WEIGHTED + WORD-BONUS + NORMALIZED)
    # ------------------------------------------------------------------
    def find_nearest_diseases(self, symptoms: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        input_symptoms = self.find_symptom_uris(symptoms)
        if not input_symptoms:
            return []

        disease_roles = self._disease_symptoms_cache
        results: List[Dict[str, Any]] = []

        for disease, role_sets in disease_roles.items():
            primary = role_sets["primary"]
            secondary = role_sets["secondary"]
            rare = role_sets["rare"]

            hit_primary = input_symptoms & primary
            hit_secondary = input_symptoms & secondary
            hit_rare = input_symptoms & rare

            if not (hit_primary or hit_secondary or hit_rare):
                continue

            # Matched points (role weight * word multiplier)
            matched_points = 0.0
            for s in hit_rare:
                matched_points += self.WEIGHT_RARE * self._symptom_word_multiplier(s)
            for s in hit_primary:
                matched_points += self.WEIGHT_PRIMARY * self._symptom_word_multiplier(s)
            for s in hit_secondary:
                matched_points += self.WEIGHT_SECONDARY * self._symptom_word_multiplier(s)

            # Max points for disease (normalization; role weight * word multiplier for ALL disease symptoms)
            max_points = 0.0
            for s in rare:
                max_points += self.WEIGHT_RARE * self._symptom_word_multiplier(s)
            for s in primary:
                max_points += self.WEIGHT_PRIMARY * self._symptom_word_multiplier(s)
            for s in secondary:
                max_points += self.WEIGHT_SECONDARY * self._symptom_word_multiplier(s)

            similarity = (matched_points / max_points) if max_points > 0 else 0.0
            matched_symptoms_all = hit_primary | hit_secondary | hit_rare

            results.append(
                {
                    "disease_uri": str(disease),
                    "disease_name": self.get_disease_label(disease),
                    "matched_symptoms": sorted(self.get_symptom_label(s) for s in matched_symptoms_all),
                    "match_count": len(matched_symptoms_all),
                    "matched_points": matched_points,
                    "max_points": max_points,
                    "similarity_score": similarity,
                    "matched_primary": sorted(self.get_symptom_label(s) for s in hit_primary),
                    "matched_secondary": sorted(self.get_symptom_label(s) for s in hit_secondary),
                    "matched_rare": sorted(self.get_symptom_label(s) for s in hit_rare),
                    "total_primary": len(primary),
                    "total_secondary": len(secondary),
                    "total_rare": len(rare),
                }
            )

        results.sort(key=lambda r: (-r["similarity_score"], -r["matched_points"], -r["match_count"], r["disease_name"]))
        return results[:top_k] if top_k else results

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_primary_symptoms(self, disease_name: str) -> List[str]:
        target = self.normalize(disease_name)
        for disease in self.graph.subjects(RDF.type, self.EX.Disease):
            label = self.normalize(self.get_disease_label(disease))
            if target == label:
                return sorted({self.get_symptom_label(s) for s in self.graph.objects(disease, self.EX.hasPrimarySymptom)})
        return []

    def get_wikidata_id(self, disease_name: str) -> Optional[str]:
        target = self.normalize(disease_name)
        for disease in self.graph.subjects(RDF.type, self.EX.Disease):
            label = self.normalize(self.get_disease_label(disease))
            if target == label:
                for prop in (OWL.equivalentClass, OWL.sameAs):
                    for eq in self.graph.objects(disease, prop):
                        s_eq = str(eq)
                        if "wikidata.org/entity/" in s_eq:
                            return s_eq.split("/")[-1]
        return None

    def get_all_symptoms(self) -> List[str]:
        symptom_types = {self.EX.Symptom}
        for subclass in self.graph.subjects(RDFS.subClassOf, self.EX.Symptom):
            symptom_types.add(subclass)

        return sorted(
            {
                self.get_symptom_label(s)
                for symptom_type in symptom_types
                for s in self.graph.subjects(RDF.type, symptom_type)
            }
        )

    # ==================================================================
    # ===================== SPARQL QUERY UTILITIES =====================
    # ==================================================================

    @staticmethod
    def _symps_to_values(symps_list: List[str]) -> str:
        """
        Convert ["wd:Q38933", "wd:Q9690"] -> "wd:Q38933 wd:Q9690"
        Assumes caller passes valid prefixed names.
        """
        return " ".join(symps_list)

    def query_topk_diseases_by_score(
        self,
        symps_list: List[str],
        top_k_results: int = 5,
        exclude_label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        SPARQL QUERY 1:
        Top-k diseases by combined score.
        Optionally exclude a disease label (case-insensitive).
        Input symptoms must be wd:Q... etc.
        """
        input_symps = self._symps_to_values(symps_list)
        exclude = (exclude_label or "").strip()

        exclude_filter = ""
        if exclude:
            # safer than interpolating raw user text in many places: still, keep it simple
            # compares label string case-insensitively
            exclude_filter = f'FILTER(LCASE(STR(?label)) != LCASE("{exclude}"))'

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

  ### MATCHED COUNT ###
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

  ### JACCARD ###
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

  ### IDF RAW ###
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

  ### IDF MAX ###
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

  ### PRIMARY SYMPTOMS MENTIONED ###
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

  ### TOTAL SCORE ###
  BIND( (0.6 * ?idfNorm + 0.2 * ?jaccard + 0.2 * ?coverage) AS ?baseScore )
  BIND( (?baseScore * ?coreCoeff + 0.10 * ?catNorm) AS ?finalScore )
}}
GROUP BY ?disease ?label ?catNorm ?baseScore ?finalScore
ORDER BY DESC(?finalScore)
LIMIT {int(top_k_results)}
"""
        rows = []
        for row in self.graph.query(q):
            # row: (?disease ?label ?catNorm ?baseScore ?finalScore)
            rows.append(
                {
                    "disease_uri": str(row[0]),
                    "label": str(row[1]),
                    "catNorm": float(row[2]) if row[2] is not None else None,
                    "baseScore": float(row[3]) if row[3] is not None else None,
                    "finalScore": float(row[4]) if row[4] is not None else None,
                }
            )
        return rows

    def query_disease_wiki_info(self, disease_iri: str) -> Optional[Dict[str, str]]:
        """
        SPARQL QUERY 2:
        Give wiki info on 1 disease (must be full IRI string).
        """
        disease = f"<{disease_iri.strip().strip('<>')}>"
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
        for row in self.graph.query(q):
            return {
                "label": str(row[0]),
                "explanation": str(row[1]),
                "seeDoctor": str(row[2]),
                "treatment": str(row[3]),
            }
        return None

    def query_matching_symptoms(self, disease_iri: str, symps_list: List[str]) -> List[str]:
        """
        SPARQL QUERY 3a:
        Matching symptoms between disease and user list.
        Returns list of symptom labels.
        """
        disease = f"<{disease_iri.strip().strip('<>')}>"
        input_symps = self._symps_to_values(symps_list)

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
        return [str(row[0]) for row in self.graph.query(q)]

    def query_missing_symptoms(self, disease_iri: str, symps_list: List[str]) -> List[str]:
        """
        SPARQL QUERY 3b:
        Missing symptoms: disease symptoms MINUS user list.
        Returns list of symptom labels.
        """
        disease = f"<{disease_iri.strip().strip('<>')}>"
        input_symps = self._symps_to_values(symps_list)

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
        return [str(row[0]) for row in self.graph.query(q)]


# ----------------------------------------------------------------------
# CLI / Example usage
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttl", default="ontology/databaseV72.ttl", help="Path to TTL file.")
    ap.add_argument(
        "--mode",
        default="python",
        choices=["python", "q1", "q2", "q3"],
        help="python = Python scorer; q1/q2/q3 = run SPARQL Query 1/2/3.",
    )

    # Python scorer inputs (labels)
    ap.add_argument(
        "--symptoms",
        default="chest pain;cough;fever;pain",
        help="Semicolon-separated symptom LABELS for Python scorer mode.",
    )
    ap.add_argument("--topk", type=int, default=5, help="Top K diseases")

    # SPARQL inputs (wd:Q...)
    ap.add_argument(
        "--wd_symps",
        default="wd:Q9690 wd:Q38933 wd:Q81938 wd:Q186889 wd:Q127076",
        help="Space-separated WD symptom IDs (prefixed), e.g. 'wd:Q38933 wd:Q9690'. Used by q1/q3.",
    )
    ap.add_argument("--exclude_label", default="", help="Exclude disease label in Query 1 (optional).")
    ap.add_argument(
        "--disease_iri",
        default="http://www.wikidata.org/entity/Q83319",
        help="Disease IRI for q2/q3 (full IRI, no <> needed).",
    )

    args = ap.parse_args()

    ttl_path = Path(args.ttl)
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL not found: {ttl_path.resolve()}")

    finder = RDFDiseaseFinder(str(ttl_path))

    if args.mode == "python":
        symptoms = [s.strip() for s in args.symptoms.split(";") if s.strip()]
        print(f"\n[PYTHON] Input symptoms (labels): {symptoms}\n")
        results = finder.find_nearest_diseases(symptoms, top_k=args.topk)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['disease_name']}")
            print(f"   Similarity: {r['similarity_score']:.2%}")
            print(f"   Matched: {', '.join(r['matched_symptoms'])}")
            print(f"   Points: {r['matched_points']:.2f}/{r['max_points']:.2f}")
            if r["matched_primary"]:
                print(f"   Primary hits:   {', '.join(r['matched_primary'])}")
            if r["matched_secondary"]:
                print(f"   Secondary hits: {', '.join(r['matched_secondary'])}")
            if r["matched_rare"]:
                print(f"   Rare hits:      {', '.join(r['matched_rare'])}")
            print()

    elif args.mode == "q1":
        wd_symps = [s.strip() for s in args.wd_symps.split() if s.strip()]
        print(f"\n[SPARQL Q1] Input symptoms (wd:): {wd_symps}")
        if args.exclude_label:
            print(f"[SPARQL Q1] Excluding disease label: {args.exclude_label}")
        rows = finder.query_topk_diseases_by_score(
            symps_list=wd_symps,
            top_k_results=args.topk,
            exclude_label=args.exclude_label or None,
        )
        if not rows:
            print("(no results)")
            return
        for i, r in enumerate(rows, 1):
            print(f"{i}. Disease: {r['label']}  Score: {r['finalScore']:.3f}")
            print(f"   URI: {r['disease_uri']}")

    elif args.mode == "q2":
        print(f"\n[SPARQL Q2] Disease IRI: {args.disease_iri}")
        info = finder.query_disease_wiki_info(args.disease_iri)
        if not info:
            print("(no info found)")
            return
        print(f"Disease: {info['label']}")
        print(f"Explanation: {info['explanation']}")
        print(f"Visit to the doctor: {info['seeDoctor']}")
        print(f"Treatment: {info['treatment']}")

    elif args.mode == "q3":
        wd_symps = [s.strip() for s in args.wd_symps.split() if s.strip()]
        print(f"\n[SPARQL Q3] Disease IRI: {args.disease_iri}")
        print(f"[SPARQL Q3] Input symptoms (wd:): {wd_symps}")

        matched = finder.query_matching_symptoms(args.disease_iri, wd_symps)
        missing = finder.query_missing_symptoms(args.disease_iri, wd_symps)

        print("\nMatched symptoms:")
        for s in matched:
            print(f"- {s}")

        print("\nMissing symptoms:")
        for s in missing:
            print(f"- {s}")


if __name__ == "__main__":
    main()
