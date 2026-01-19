import joblib
import sys
from pathlib import Path
from typing import List, Dict, Any


# ------------------------------------------------------------
# Project path setup
# ------------------------------------------------------------
base_dir = Path(__file__).parent.resolve()
sys.path.append(str(base_dir))

from reasoning.reasoning_engine import ReasoningEngine 
from reasoning.rdf_disease_finder import RDFDiseaseFinder
from rag.rag_engine import RAGExplainer
from reasoning.wikidata_client import WikidataClient
from reasoning.emergency_reasoner import triage_case
from reasoning import symptom_matcher


# ------------------------------------------------------------
# Symptom extraction via symptom_matcher.py (TF-IDF cosine)
# ------------------------------------------------------------
def extract_symptoms_with_matcher(
    text: str,
    rdf_graph,
    top_k: int = 15,
    threshold: float = 0.10,
) -> List[str]:
    """
    Uses TF-IDF cosine similarity against KG symptom labels (symptom_matcher.py).
    Returns symptom LABELS (strings), because rdf_disease_finder.find_nearest_diseases expects labels.
    """
    matches = symptom_matcher.match_symptoms(rdf_graph, text, top_k=top_k)
    good = symptom_matcher.symptoms_above_threshold(matches, threshold=threshold)

    # Deduplicate while keeping order
    seen = set()
    out: List[str] = []
    for m in good:
        lbl = str(m.get("label", "")).strip()
        if not lbl:
            continue
        key = lbl.lower()
        if key not in seen:
            out.append(lbl)
            seen.add(key)

    return out


# ------------------------------------------------------------
# Load components (same idea as your older main)
# ------------------------------------------------------------
def load_components(base_path: Path):
    """Load all models and data."""
    print("Loading components...", file=sys.stderr)

    model_path = base_path / "models" / "classifier.joblib"
    rdf_path = base_path / "ontology" / "databaseV7.ttl"
    docs_path = base_path / "rag" / "docs"

    components: Dict[str, Any] = {}

    # Classifier
    try:
        components["classifier"] = joblib.load(model_path)
    except Exception as e:
        print(f"Warning: Could not load classifier: {e}", file=sys.stderr)
        components["classifier"] = None

    # Reasoner (engine3)
    try:
        components["reasoner"] = ReasoningEngine()
    except Exception as e:
        print(f"Warning: Could not init ReasoningEngine: {e}", file=sys.stderr)
        components["reasoner"] = None

    # RAG explainer
    try:
        components["explainer"] = RAGExplainer(docs_path=str(docs_path))
    except Exception as e:
        print(f"Warning: Could not load RAG explainer: {e}", file=sys.stderr)
        components["explainer"] = None

    # RDF finder
    try:
        if not rdf_path.exists():
            ttls = list((base_path / "ontology").glob("*.ttl"))
            if ttls:
                rdf_path = ttls[0]
                print(f"Version 7 database not found, using {rdf_path.name}", file=sys.stderr)

        components["rdf_finder"] = RDFDiseaseFinder(str(rdf_path))
        print(f"RDF graph loaded from: {rdf_path.name}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading RDF file: {e}", file=sys.stderr)
        components["rdf_finder"] = None

    # Wikidata client
    components["wikidata"] = WikidataClient()

    # Keep this for compatibility/logging (not used for extraction anymore)
    if components.get("rdf_finder"):
        try:
            components["all_symptoms"] = components["rdf_finder"].get_all_symptoms()
            print(f"Loaded {len(components['all_symptoms'])} symptoms from Knowledge Graph.", file=sys.stderr)
        except Exception as e:
            print(f"Error fetching symptoms from KG: {e}", file=sys.stderr)
            components["all_symptoms"] = []
    else:
        components["all_symptoms"] = []

    return components


# ------------------------------------------------------------
# Diagnosis pipeline
# ------------------------------------------------------------
def run_diagnosis(text: str, components: Dict[str, Any]):
    """Run the full diagnosis pipeline on the input text."""

    print("\n" + "=" * 70)
    print("Disease Prediction System Results")
    print("=" * 70)
    print(f"Input: {text}")

    # 1) Extract Symptoms (NEW: TF-IDF matcher)
    rdf_finder = components.get("rdf_finder")
    if not rdf_finder:
        print("RDF Finder not initialized.")
        return

    symptoms = extract_symptoms_with_matcher(
        text=text,
        rdf_graph=rdf_finder.graph,
        top_k=15,
        threshold=0.10,
    )

    print(f"Extracted Symptoms: {', '.join(symptoms) if symptoms else 'None found'}\n")

    if not symptoms:
        print("No symptoms identified. Please provide more specific details.")
        return

    # 2) RDF Search (same as older main output)
    nearest_diseases = []

    print("-" * 30 + " RDF Knowledge Graph " + "-" * 30)
    try:
        nearest_diseases = rdf_finder.find_nearest_diseases(symptoms, top_k=3, use_jaccard=False)
        if nearest_diseases:
            for i, disease in enumerate(nearest_diseases, 1):
                print(f"{i}. {disease['disease_name']}")
                print(f"   Confidence: {disease['similarity_score']:.2%}")
                print(f"   Matched: {', '.join(disease['matched_symptoms'])}")
                print(f"   Coverage: {disease['match_count']}/{disease['total_input_symptoms']}")
                print()
        else:
            print("No diseases found matching the symptoms in the Knowledge Graph.")
    except Exception as e:
        print(f"Error querying KG: {e}")

    # 3) Hybrid Reasoning (Fusion) — keep same style
    classifier = components.get("classifier")
    reasoner = components.get("reasoner")
    print("-" * 30 + " Hybrid Reasoning " + "-" * 30)

    final_result = None

    if classifier and reasoner:
        try:
            probs = classifier.predict_proba([text])[0]
            labels = classifier.classes_
            top_idx = probs.argmax()

            ml_prediction = {
                "disease_id": labels[top_idx],
                "score": float(probs[top_idx]),
            }

            # Fuse results
            # NOTE: to keep old behavior/format, we pass nearest_diseases like before.
            # reasoning_engine3 can compute its own KG too, but this matches old flow.
            final_result = reasoner.fuse_results(
                ml_prediction=ml_prediction,
                rdf_candidates=nearest_diseases,
                user_symptoms=symptoms,
                rdf_finder=rdf_finder,
                ttl_path=str(base_dir / "ontology" / "databaseV7.ttl"),  # optional; engine3 uses it if needed
                top_k=5,
                symptom_match_threshold=0.10,
            )

            print(f"Final Prediction: {final_result['disease']}")
            print(f"Confidence Score: {final_result['final_score']:.2%}")

            if final_result.get("is_fallback"):
                print("Note: Result based on Knowledge Graph due to low ML confidence.")
            else:
                print(f"Base ML Score:    {final_result['original_score']:.2%}")

            print("\nReasoning Trace:")
            for reason in final_result.get("reasoning", []):
                try:
                    print(f"  {reason}")
                except UnicodeEncodeError:
                    print(f"  {reason.encode('ascii', 'ignore').decode()}")
            print()

        except Exception as e:
            print(f"Error in reasoning engine: {e}")
    else:
        print("Classifier or Reasoner not available.")

    # 4) Live Wikidata Info (same structure)
    wikidata = components.get("wikidata")
    if final_result and final_result.get("disease"):
        print("-" * 30 + " Live Wikidata Info " + "-" * 30)
        disease_name = final_result["disease"]

        wikidata_id = None
        try:
            wikidata_id = rdf_finder.get_wikidata_id(disease_name)
        except Exception:
            wikidata_id = None

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

    # 5) Explanation (same structure)
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



# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    """
    Main execution function.
    Runs a sample diagnosis flow when the script is executed directly.
    """
    components = load_components(base_dir)

    # Sample input
    sample_text = "I have a few days of fever and cough. And I have chest pain."

    run_diagnosis(sample_text, components)


if __name__ == "__main__":
    main()
