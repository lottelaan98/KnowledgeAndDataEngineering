import joblib
import sys
from pathlib import Path
from typing import List, Dict, Any
import re


base_dir = Path(__file__).parent.resolve()
sys.path.append(str(base_dir))

from reasoning.reasoning_engine import ReasoningEngine
from reasoning.rdf_disease_finder_with_queries import RDFDiseaseFinder
from rag.rag_engine import RAGExplainer
from reasoning.wikidata_client import WikidataClient
from reasoning.symptom_matcher import match_symptoms, symptoms_above_threshold
from reasoning.emergency_reasoner3 import triage_case



# Common symptoms loaded dynamically from the KG

def extract_symptoms_from_text(text: str, known_symptoms: List[str]) -> List[str]:
    """
    Extract symptom keywords from user text.
    Uses regex word-boundary matching instead of substring + replace.
    """
    if not known_symptoms:
        return []

    # Normalize input text: lowercase + keep spaces
    text_lower = text.lower()
    text_lower = re.sub(r"[^a-z0-9\s]", " ", text_lower)
    text_lower = re.sub(r"\s+", " ", text_lower).strip()

    found = []
    found_set = set()

    # 1) Match known symptoms (prefer longer phrases first)
    for symptom in sorted(known_symptoms, key=len, reverse=True):
        s = symptom.lower().strip()
        if not s:
            continue

        # Make spaces flexible: "skin rash" matches "skin   rash"
        pattern = r"\b" + re.escape(s).replace(r"\ ", r"\s+") + r"\b"

        if re.search(pattern, text_lower):
            if s not in found_set:
                found.append(s)
                found_set.add(s)

    # 2) Extra phrase -> canonical symptom mapping (optional)
    synonyms = {
        "loose stool": "diarrhea",
        "stomach ache": "abdominal pain",
        "tummy ache": "abdominal pain",
        "high temp": "fever",
        "throwing up": "vomiting",
        "shitting": "diarrhea",
        "itchy": "itch",  
    }

    for phrase, canonical_symptom in synonyms.items():
        p = phrase.lower().strip()
        pattern = r"\b" + re.escape(p).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text_lower):
            c = canonical_symptom.lower()
            if c not in found_set:
                found.append(c)
                found_set.add(c)

    return found


def load_components(base_path: Path):
    """Load all models and data."""
    print("Loading components...", file=sys.stderr)
    
    model_path = base_path / "models" / "classifier.joblib"
    rdf_path = base_path / "ontology" / "databaseV72.ttl"
    docs_path = base_path / "rag" / "docs"

    components = {}

    try:
        components['classifier'] = joblib.load(model_path)
    except Exception as e:
        print(f"Warning: Could not load classifier: {e}", file=sys.stderr)
        components['classifier'] = None

    try:
        components['reasoner'] = ReasoningEngine()
    except Exception:
        components['reasoner'] = None

    try:
        components['explainer'] = RAGExplainer(docs_path=str(docs_path))
    except Exception as e:
        print(f"Warning: Could not load RAG explainer: {e}", file=sys.stderr)
        components['explainer'] = None
        
    try:
        # Check if version 2 exists, otherwise fallback or error
        if not rdf_path.exists():
            # Try finding any ttl in ontology
            ttls = list((base_path / "ontology").glob("*.ttl"))
            if ttls:
                rdf_path = ttls[0]
                print(f"Version 2 database not found, using {rdf_path.name}", file=sys.stderr)
        
        components['rdf_finder'] = RDFDiseaseFinder(str(rdf_path))
        print(f"RDF graph loaded from: {rdf_path.name}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading RDF file: {e}", file=sys.stderr)
        components['rdf_finder'] = None

    # Load Wikidata Client
    components['wikidata'] = WikidataClient()

    # Pre-fetch all recognized symptoms from the ontology
    if components.get('rdf_finder'):
        try:
            components['all_symptoms'] = components['rdf_finder'].get_all_symptoms()
            print(f"Loaded {len(components['all_symptoms'])} symptoms from Knowledge Graph.", file=sys.stderr)
        except Exception as e:
            print(f"Error fetching symptoms from KG: {e}", file=sys.stderr)
            components['all_symptoms'] = []
    else:
        components['all_symptoms'] = []

    return components

def run_diagnosis(text: str, components: Dict[str, Any]):
    """Run the full diagnosis pipeline on the input text."""

    print("\n" + "=" * 70)
    print("Disease Prediction System Results")
    print("=" * 70)
    print(f"Input: {text}")

    # ------------------------------------------------------------------
    # 1. Extract Symptoms (SIMILARITY-BASED via reasoning.symptom_matcher)
    # ------------------------------------------------------------------
    rdf_finder = components.get("rdf_finder")
    if not rdf_finder:
        print("RDF Finder not initialized.")
        return

    # rdflib Graph is inside rdf_finder
    g = getattr(rdf_finder, "graph", None)
    if g is None:
        print("RDF graph not available in rdf_finder.")
        return

    # Import locally to avoid breaking your program if module path changes
    from reasoning.symptom_matcher import match_symptoms, symptoms_above_threshold

    try:
        # Get many candidates, then filter by threshold
        matches = match_symptoms(g, text, top_k=50)
        matches = symptoms_above_threshold(matches, threshold=0.1)

        # Convert to labels for your downstream pipeline (expects List[str])
        symptoms = [m["label"].lower() for m in matches]

        print("\n=== Matched Symptoms (cosine TF-IDF) ===")
        if matches:
            for m in matches:
                print(f"- {m['label']} ({m['score']:.3f})")
        else:
            print("(none above threshold)")

        print(f"\nExtracted Symptoms (labels): {', '.join(symptoms) if symptoms else 'None found'}\n")

    except Exception as e:
        print(f"Error during symptom matching: {e}")
        return

    if not symptoms:
        print("No symptoms identified. Please provide more specific details.")
        return

    # ------------------------------------------------------------------
    # 2. RDF Search
    # ------------------------------------------------------------------
    nearest_diseases = []

    print("-" * 30 + " RDF Knowledge Graph " + "-" * 30)
    
    try:
        nearest_diseases = rdf_finder.find_nearest_diseases(symptoms, top_k=3)
        if nearest_diseases:
            for i, disease in enumerate(nearest_diseases, 1):
                print(f"{i}. {disease['disease_name']}")
                print(f"   Confidence: {disease['similarity_score']:.2%}")
                print(f"   Points: {disease['matched_points']}/{disease['max_points']}")
                print(f"   Matched: {', '.join(disease['matched_symptoms'])}")

                if disease.get("matched_primary"):
                    print(f"   Primary hits:   {', '.join(disease['matched_primary'])}")
                if disease.get("matched_secondary"):
                    print(f"   Secondary hits: {', '.join(disease['matched_secondary'])}")
                if disease.get("matched_rare"):
                    print(f"   Rare hits:      {', '.join(disease['matched_rare'])}")

                print()
        else:
            print("No diseases found matching the symptoms in the Knowledge Graph.")
    except Exception as e:
        print(f"Error querying KG: {e}")

    # ------------------------------------------------------------------
    # 3. Hybrid Reasoning (Fusion)
    # ------------------------------------------------------------------
    classifier = components.get("classifier")
    reasoner = components.get("reasoner")
    print("-" * 30 + " Hybrid Reasoning " + "-" * 30)

    final_result = None  # IMPORTANT: avoid UnboundLocalError later

    if classifier and reasoner:
        try:
            probs = classifier.predict_proba([text])[0]
            labels = classifier.classes_
            top_idx = probs.argmax()
            ml_prediction = {"disease_id": labels[top_idx], "score": float(probs[top_idx])}

            final_result = reasoner.fuse_results(
                ml_prediction=ml_prediction,
                rdf_candidates=nearest_diseases,
                user_symptoms=symptoms,
                rdf_finder=rdf_finder,
            )

            print(f"Final Prediction: {final_result['disease']}")
            print(f"Confidence Score: {final_result['final_score']:.2%}")

            if final_result.get("is_fallback"):
                print("Note: Result based on Knowledge Graph due to low ML confidence.")
            else:
                print(f"Base ML Score:    {final_result['original_score']:.2%}")

            print("\nReasoning Trace:")
            for reason in final_result["reasoning"]:
                try:
                    print(f"  {reason}")
                except UnicodeEncodeError:
                    print(f"  {reason.encode('ascii', 'ignore').decode()}")
            print()

        except Exception as e:
            print(f"Error in reasoning engine: {e}")
    else:
        print("Classifier or Reasoner not available.")

    # ------------------------------------------------------------------
    # 4. Live Wikidata Info
    # ------------------------------------------------------------------
    wikidata = components.get("wikidata")
    if final_result and final_result.get("disease"):
        print("-" * 30 + " Live Wikidata Info " + "-" * 30)
        disease_name = final_result["disease"]
        disease_uri = None
        if nearest_diseases:
            disease_uri = nearest_diseases[0].get("disease_uri")

        wikidata_id = None
        if disease_uri and "wikidata.org/entity/" in disease_uri:
            wikidata_id = disease_uri.rsplit("/", 1)[-1]

        if wikidata_id and wikidata:
            print(f"Fetching data for {disease_name} ({wikidata_id})...")
            info = wikidata.fetch_disease_info(wikidata_id)
            if info:
                print(f"Description: {info.get('description')}")
                print(f"Wikipedia:   {info.get('wikipedia_url')}")
                if info.get("image_url"):
                    print(f"Image:       {info.get('image_url')}")
            else:
                print("No additional info found on Wikidata.")
        else:
            print("No Wikidata ID found in ontology.")

    # ------------------------------------------------------------------
    # 5. Explanation
    # ------------------------------------------------------------------
    explainer = components.get("explainer")
    if nearest_diseases and explainer:
        print("-" * 30 + " Explanation " + "-" * 30)
        top_disease = nearest_diseases[0]
        try:
            explanation = explainer.explain(
                symptoms=text,
                disease=top_disease["disease_name"],
                confidence=top_disease["similarity_score"],
            )
            print(f"\nExplanation for {top_disease['disease_name']}:")
            print(explanation)
        except Exception as e:
            print(f"Could not generate explanation: {e}")

    # ------------------------------------------------------------------
    # 6. TRIAGE (KG rules + text cosine)
    # ------------------------------------------------------------------
    print("-" * 30 + " Triage " + "-" * 30)

    try:
        # rdflib Graph already available as `g`
        # disease_iri: if your rdf_finder can provide it, use it; else None
        disease_iri = None
        if final_result and final_result.get("disease"):
            # If this returns a Wikidata QID, you can turn it into an IRI:
            # e.g., "Q179945" -> "http://www.wikidata.org/entity/Q179945"
            qid = rdf_finder.get_wikidata_id(final_result["disease"])
            if qid:
                disease_iri = f"http://www.wikidata.org/entity/{qid}"

        triage_result = triage_case(
            g=g,
            user_text=text,
            symptom_iris=[m["uri"] for m in matches if m.get("uri")],          # TODO: upgrade to real IRIs (see next section)
            disease_iri=nearest_diseases[0].get("disease_uri") if nearest_diseases else None,
            temperatureC=None,
            systolicBP=None,
            painScale=None,
        )

        print("Triage final:", triage_result["final"], "| source:", triage_result["source"])
        if triage_result.get("seeDoctorText"):
            print("seeDoctor:", triage_result["seeDoctorText"])

        # optional debug
        print("KG pred:", triage_result["kg"].pred, "score_sum:", triage_result["kg"].score_sum)
        print("Text pred:", triage_result["text"]["pred"], triage_result["text"]["scores"])

    except Exception as e:
        print(f"Error running triage engine: {e}")


def main():
    """
    Main execution function.
    Runs a sample diagnosis flow when the script is executed directly.
    """
    components = load_components(base_dir)
    
    sample_text = "I’ve had a runny nose and nasal congestion for several days, with sneezing and a sore throat. I’m also coughing and feeling fatigued."
    
    run_diagnosis(sample_text, components)

if __name__ == "__main__":
    main()
