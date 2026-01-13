"""
RDF-based disease finder aligned with the refactored medical ontology.
Supports symptom-role aware querying and similarity scoring.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

from rdflib import Graph, Namespace, RDF, OWL, URIRef
from rdflib.namespace import SKOS, RDFS


class RDFDiseaseFinder:
    def __init__(self, rdf_path: str):
        self.graph = Graph()
        self.graph.parse(rdf_path, format="turtle")

        # Hard-bind namespace (stop guessing)
        self.EX = Namespace("http://uu.nl/medical/")
        self.graph.bind("ex", self.EX)

        # Caches
        self._symptom_label_cache: Dict[str, str] = {}
        self._disease_label_cache: Dict[str, str] = {}

        # Sanity checks (fail fast)
        if not any(self.graph.triples((None, RDF.type, self.EX.Disease))):
            raise RuntimeError("No ex:Disease instances found. Check ontology or namespace.")

        # Cache content
        self._disease_symptoms_cache = self.get_all_disease_symptoms()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"[-_]+", " ", text.lower().strip())

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

        # Fallback: local name
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

        # Get all symptom subclasses (e.g., ex:SystemicSymptom, ex:SkinSymptom, etc.)
        symptom_types = {self.EX.Symptom}
        for subclass in self.graph.subjects(RDFS.subClassOf, self.EX.Symptom):
            symptom_types.add(subclass)

        # Find symptoms of any symptom type
        for symptom_type in symptom_types:
            for symptom in self.graph.subjects(RDF.type, symptom_type):
                label = self.normalize(self.get_symptom_label(symptom))
                for s in normalized_inputs:
                    if s == label or s in label or label in s:
                        matches.add(symptom)

        return matches

    # ------------------------------------------------------------------
    # Disease → symptoms (ALL ROLES)
    # ------------------------------------------------------------------

    def get_all_disease_symptoms(self) -> Dict[URIRef, Set[URIRef]]:
        disease_symptoms = defaultdict(set)

        role_props = (
            self.EX.hasPrimarySymptom,
            self.EX.hasSecondarySymptom,
            self.EX.hasComplication,
        )

        for disease in self.graph.subjects(RDF.type, self.EX.Disease):
            for prop in role_props:
                for symptom in self.graph.objects(disease, prop):
                    disease_symptoms[disease].add(symptom)

        return disease_symptoms

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_nearest_diseases(
        self,
        symptoms: List[str],
        top_k: Optional[int] = None,
        use_jaccard: bool = True,
    ) -> List[Dict]:

        input_symptoms = self.find_symptom_uris(symptoms)
        if not input_symptoms:
            return []

        # Use cached index
        disease_symptoms = self._disease_symptoms_cache
        results = []

        for disease, ds in disease_symptoms.items():
            intersection = input_symptoms & ds
            if not intersection:
                continue

            union = input_symptoms | ds
            score = (
                len(intersection) / len(union)
                if use_jaccard and union
                else len(intersection) / len(input_symptoms)
            )

            results.append({
                "disease_uri": disease,
                "disease_name": self.get_disease_label(disease),
                "matched_symptoms": sorted(self.get_symptom_label(s) for s in intersection),
                "match_count": len(intersection),
                "similarity_score": score,
                "total_disease_symptoms": len(ds),
                "total_input_symptoms": len(input_symptoms),
            })

        results.sort(
            key=lambda r: (-r["similarity_score"], -r["match_count"], r["disease_name"])
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
                for prop in (
                    self.EX.hasPrimarySymptom,
                    self.EX.hasSecondarySymptom,
                    self.EX.hasComplication,
                ):
                    for s in self.graph.objects(disease, prop):
                        symptoms.add(self.get_symptom_label(s))
                return sorted(symptoms)

        return []

    def get_primary_symptoms(self, disease_name: str) -> List[str]:
        """
        Get only the primary symptoms for a specific disease.
        Used for validation/sanity checks.
        """
        target = self.normalize(disease_name)
        
        for disease in self.graph.subjects(RDF.type, self.EX.Disease):
            label = self.normalize(self.get_disease_label(disease))
            if target == label: # Strict match for sanity check
                symptoms = set()
                for s in self.graph.objects(disease, self.EX.hasPrimarySymptom):
                    symptoms.add(self.get_symptom_label(s))
                return sorted(symptoms)
        return []

    def get_wikidata_id(self, disease_name: str) -> Optional[str]:
        """
        Extract the Wikidata Q-ID for a given disease name from the ontology.
        Uses the owl:equivalentClass or owl:sameAs relationship.
        """
        target = self.normalize(disease_name)
        
        for disease in self.graph.subjects(RDF.type, self.EX.Disease):
            label = self.normalize(self.get_disease_label(disease))
            if target == label:
                # Look for owl:equivalentClass or owl:sameAs that points to wd namespace
                for prop in (OWL.equivalentClass, OWL.sameAs):
                    for eq in self.graph.objects(disease, prop):
                        # Check if the URI starts with wikidata entity prefix
                        s_eq = str(eq)
                        if "wikidata.org/entity/" in s_eq:
                            return s_eq.split("/")[-1]
        return None

    def get_all_symptoms(self) -> List[str]:
        symptom_types = {self.EX.Symptom}
        for subclass in self.graph.subjects(RDFS.subClassOf, self.EX.Symptom):
            symptom_types.add(subclass)
            
        return sorted({
            self.get_symptom_label(s)
            for symptom_type in symptom_types
            for s in self.graph.subjects(RDF.type, symptom_type)
        })


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

def main():
    base_dir = Path(__file__).parent.parent
    rdf_path = base_dir / "ontology" / "version 2 database.ttl"

    finder = RDFDiseaseFinder(str(rdf_path))

    symptoms = ["fever", "headache"]
    print(f"\nInput symptoms: {symptoms}\n")

    results = finder.find_nearest_diseases(symptoms, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['disease_name']}")
        print(f"   Similarity: {r['similarity_score']:.2%}")
        print(f"   Matched: {', '.join(r['matched_symptoms'])}\n")


if __name__ == "__main__":
    main()
