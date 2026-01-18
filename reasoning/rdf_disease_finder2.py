"""
RDF-based disease finder aligned with the refactored medical ontology.
Supports symptom-role aware querying and similarity scoring.

CHANGES:
- Role-aware weighting:
    * Rare symptom = 3 points
    * Primary symptom = 2 points
    * Secondary symptom = 1 point
- Extra weight for multi-word symptom matches:
    * weight multiplier = 1 + WORD_BONUS_ALPHA * (num_words - 1)
      (so 1 word => 1.0x, 2 words => 1.0+alpha, 3 words => 1.0+2alpha, etc.)
- Normalized disease score so diseases with many symptoms are not penalized:
    similarity_score = (weighted_matched_points) / (max_possible_weighted_points_for_disease)
  where max_possible_weighted_points_for_disease is the sum of weighted role points for ALL symptoms of that disease.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

from rdflib import Graph, Namespace, RDF, OWL, URIRef
from rdflib.namespace import SKOS, RDFS


class RDFDiseaseFinder:
    # Role weights
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

        self.EX = Namespace("http://example.org/med#")
        self.graph.bind("ex", self.EX)

        self._symptom_label_cache: Dict[str, str] = {}
        self._disease_label_cache: Dict[str, str] = {}

        if not any(self.graph.triples((None, RDF.type, self.EX.Disease))):
            raise RuntimeError("No ex:Disease instances found. Check ontology or namespace.")

        # role-aware disease symptom index
        self._disease_symptoms_cache = self.get_all_disease_symptoms()

    # ------------------------------------------------------------------
    # Normalization
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
        """
        Multi-word symptoms get a higher multiplier.
        Uses the symptom label from KG.
        """
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
                if not label.language or label.language == "en":
                    cache[uri_str] = str(label)
                    return cache[uri_str]

        cache[uri_str] = uri_str.split("/")[-1]
        return cache[uri_str]

    def get_symptom_label(self, uri: URIRef) -> str:
        return self._get_label(uri, self._symptom_label_cache)

    def get_disease_label(self, uri: URIRef) -> str:
        return self._get_label(uri, self._disease_label_cache)

    # ------------------------------------------------------------------
    # Symptom lookup
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
    # Disease → symptoms (ROLE-AWARE)
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
            for s in self.graph.objects(disease, self.EX.hasRareSymptom):
                disease_symptoms[disease]["rare"].add(s)

        return disease_symptoms

    # ------------------------------------------------------------------
    # Similarity search (ROLE-WEIGHTED + WORD-BONUS + NORMALIZED)
    # ------------------------------------------------------------------

    def find_nearest_diseases(
        self,
        symptoms: List[str],
        top_k: Optional[int] = None,
    ) -> List[Dict]:

        input_symptoms = self.find_symptom_uris(symptoms)
        if not input_symptoms:
            return []

        disease_roles = self._disease_symptoms_cache
        results = []

        for disease, role_sets in disease_roles.items():
            primary = role_sets["primary"]
            secondary = role_sets["secondary"]
            rare = role_sets["rare"]

            hit_primary = input_symptoms & primary
            hit_secondary = input_symptoms & secondary
            hit_rare = input_symptoms & rare

            if not (hit_primary or hit_secondary or hit_rare):
                continue

            # --- matched weighted points (role weight * word multiplier) ---
            matched_points = 0.0
            for s in hit_rare:
                matched_points += self.WEIGHT_RARE * self._symptom_word_multiplier(s)
            for s in hit_primary:
                matched_points += self.WEIGHT_PRIMARY * self._symptom_word_multiplier(s)
            for s in hit_secondary:
                matched_points += self.WEIGHT_SECONDARY * self._symptom_word_multiplier(s)

            # --- max possible points for disease (role weight * word multiplier for ALL disease symptoms) ---
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
                    "disease_uri": disease,
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

        results.sort(
            key=lambda r: (-r["similarity_score"], -r["matched_points"], -r["match_count"], r["disease_name"])
        )

        return results[:top_k] if top_k else results

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_disease_symptoms(self, disease_name: str) -> List[str]:
        target = self.normalize(disease_name)

        for disease in self.graph.subjects(RDF.type, self.EX.Disease):
            label = self.normalize(self.get_disease_label(disease))
            if target in label:
                symptoms = set()
                for prop in (self.EX.hasPrimarySymptom, self.EX.hasSecondarySymptom, self.EX.hasRareSymptom):
                    for s in self.graph.objects(disease, prop):
                        symptoms.add(self.get_symptom_label(s))
                return sorted(symptoms)

        return []

    def get_primary_symptoms(self, disease_name: str) -> List[str]:
        target = self.normalize(disease_name)

        for disease in self.graph.subjects(RDF.type, self.EX.Disease):
            label = self.normalize(self.get_disease_label(disease))
            if target == label:
                symptoms = set()
                for s in self.graph.objects(disease, self.EX.hasPrimarySymptom):
                    symptoms.add(self.get_symptom_label(s))
                return sorted(symptoms)
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


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

def main():
    base_dir = Path(__file__).parent.parent
    rdf_path = base_dir / "ontology" / "databaseV7.ttl"

    finder = RDFDiseaseFinder(str(rdf_path))

    symptoms = ["rash", "red skin", "red, inflamed skin patches"]
    print(f"\nInput symptoms: {symptoms}\n")

    results = finder.find_nearest_diseases(symptoms, top_k=5)

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


if __name__ == "__main__":
    main()
