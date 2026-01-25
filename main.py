import joblib
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from rdflib import Graph

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
from querying import scriptV3
from UI import UI
from evaluation.dataset_statistics import compute_stats


# ------------------------------------------------------------
# Symptom extraction via symptom_matcher.py 
# ------------------------------------------------------------
def extract_symptoms_with_matcher(
    text: str,
    rdf_graph: Graph,
    top_k: int = 15,
    threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Uses TF-IDF cosine similarity against KG symptom labels (symptom_matcher.py).

    Returns:
      {
        "labels":  [..],  # symptom labels (strings)
        "iris":    [..],  # full IRIs (strings)
        "matches": [..],  # raw match dicts: {"uri","label","score"}
      }
    """
    matches = symptom_matcher.match_symptoms(rdf_graph, text, top_k=top_k)
    good = symptom_matcher.symptoms_above_threshold(matches, threshold=threshold)

    seen_lbl = set()
    labels: List[str] = []
    iris: List[str] = []

    for m in good:
        lbl = str(m.get("label", "")).strip()
        uri = str(m.get("uri", "")).strip()

        if uri:
            iris.append(uri)

        if lbl:
            key = lbl.lower()
            if key not in seen_lbl:
                labels.append(lbl)
                seen_lbl.add(key)

    # dedup IRIs, keep order
    seen_uri = set()
    iris2 = []
    for u in iris:
        if u not in seen_uri:
            iris2.append(u)
            seen_uri.add(u)

    return {"labels": labels, "iris": iris2, "matches": good}


# ------------------------------------------------------------
# Convert full URIs 
# ------------------------------------------------------------
def uris_to_prefixed(symptom_iris: List[str]) -> List[str]:
    """
    Convert:
      - http://www.wikidata.org/entity/Q123 -> wd:Q123
      - http://example.org/med#symptom/foo  -> sym:foo   
    """
    out: List[str] = []
    for uri in symptom_iris:
        uri = uri.strip()
        if not uri:
            continue

        if "wikidata.org/entity/Q" in uri:
            qid = uri.rsplit("/", 1)[-1]
            out.append(f"wd:{qid}")
            continue

        if "example.org/med#symptom/" in uri:
            local_id = uri.rsplit("/", 1)[-1]
            out.append(f"sym:{local_id}")
            continue

        # If your local symptom URIs are like http://example.org/med#bulging_blue_veins
        if "example.org/med#" in uri:
            local_id = uri.rsplit("#", 1)[-1]
            # only prefix if it looks like a symptom id
            if local_id:
                out.append(f"sym:{local_id}")
            continue

    return out


# ------------------------------------------------------------
# Load components
# ------------------------------------------------------------
def load_components(base_path: Path) -> Dict[str, Any]:
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

    # Reasoner
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

    # RDF finder + graph
    try:
        if not rdf_path.exists():
            ttls = list((base_path / "ontology").glob("*.ttl"))
            if ttls:
                rdf_path = ttls[0]
                print(f"databaseV7.ttl not found, using {rdf_path.name}", file=sys.stderr)

        components["rdf_finder"] = RDFDiseaseFinder(str(rdf_path))
        print(f"RDF graph loaded from: {rdf_path.name}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading RDF file: {e}", file=sys.stderr)
        components["rdf_finder"] = None

    components["wikidata"] = WikidataClient()
    return components


# ------------------------------------------------------------
# Diagnosis pipeline
# ------------------------------------------------------------
def run_diagnosis(text: str, components: Dict[str, Any], temperature = None, systolicBP = None, painScale = None):
    print("\n" + "=" * 70)
    print("Disease Prediction System Results")
    print("=" * 70)
    print(f"Input: {text}")

    rdf_finder = components.get("rdf_finder")
    if not rdf_finder:
        print("RDF Finder not initialized.")
        return

    g = rdf_finder.graph  # already loaded graph

    # 1) Extract symptoms via TF-IDF matcher
    extracted = extract_symptoms_with_matcher(text=text, rdf_graph=g, top_k=15, threshold=0.10)
    symptoms = extracted["labels"]
    symptom_iris = extracted["iris"]
    symptom_matches = extracted["matches"]

    print(f"Extracted Symptoms: {', '.join(symptoms) if symptoms else 'None found'}\n")

    if not symptoms:
        print("No symptoms identified. Please provide more specific details.")
        return

    # 2) KG Query pipeline (Query 1 + Query3)
    print("-" * 30 + " RDF Knowledge Graph (Query) " + "-" * 30)

    symptom_prefixed = uris_to_prefixed(symptom_iris)

    kg_candidates: List[Dict[str, Any]] = []
    try:
        rows = scriptV3.query1_topk_diseases_by_score(
            g=g,
            symps_list=symptom_prefixed,
            top_k_results=5,
            exclude_label=None,
        )

        if not rows:
            print("No diseases found (Query 1 returned empty).")
        else:
            # Optional: don't print the raw rows list (remove noisy debug)
            # print(rows)

            for i, r in enumerate(rows, 1):
                disease_name = r["label"]
                disease_iri = r["disease_uri"]
                score = float(r.get("finalScore") or 0.0)
                similarity_pct = score * 100.0

                matched_labels = scriptV3.query3_matching_symptoms(
                    g=g,
                    disease_iri=disease_iri,
                    symps_list=symptom_prefixed,
                )

                print(f"{i}. {disease_name}")
                print(f"   Disease URI: {disease_iri}")
                print(f"   Similarity: {similarity_pct:.2f}%")
                print(f"   Matched: {', '.join(matched_labels)}\n")

                kg_candidates.append(
                    {
                        "disease_name": disease_name,
                        "disease_uri": disease_iri,
                        "similarity_score": score,          # 0..1
                        "similarity_pct": similarity_pct,   # 0..100
                        "matched_symptoms": matched_labels,
                    }
                )

    except Exception as e:
        print(f"Error querying KG (query pipeline): {e}")

    # 3) Hybrid reasoning (ML + KG)
    print("-" * 30 + " Hybrid Reasoning " + "-" * 30)

    classifier = components.get("classifier")
    reasoner = components.get("reasoner")

    final_result: Optional[Dict[str, Any]] = None

    if classifier and reasoner:
        try:
            probs = classifier.predict_proba([text])[0]
            labels = classifier.classes_
            top_idx = probs.argmax()

            ml_prediction = {"disease_id": labels[top_idx], "score": float(probs[top_idx])}

            final_result = reasoner.fuse_results(
                ml_prediction=ml_prediction,
                kg_candidates=kg_candidates,
                symptom_matches=symptom_matches,
                rdf_finder=rdf_finder,
            )

            print(f"Final Prediction: {final_result['disease']}")
            print(f"Confidence Score: {final_result['final_score']:.2%}")

            if final_result.get("is_fallback"):
                print("Note: Result based on Knowledge Graph due to low ML confidence.")
            else:
                print(f"Base ML Score:    {final_result['original_score']:.2%}")

            print("\nReasoning Trace:")
            for reason in final_result.get("reasoning", []):
                print(f"  {reason}")
            print()

        except Exception as e:
            print(f"Error in reasoning engine: {e}")
    else:
        print("Classifier or Reasoner not available.")

    # 4) Live Wikidata Info (use query-based kg_candidates)
    print("-" * 30 + " Live Wikidata Info " + "-" * 30)

    wikidata = components.get("wikidata")
    if final_result and wikidata and kg_candidates:
        disease_uri = kg_candidates[0].get("disease_uri")
        wikidata_id = None
        if disease_uri and "wikidata.org/entity/Q" in disease_uri:
            wikidata_id = disease_uri.rsplit("/", 1)[-1]

        if wikidata_id:
            print(f"Fetching data for {final_result['disease']} ({wikidata_id})...")
            try:
                info = wikidata.fetch_disease_info(wikidata_id)
            except Exception as e:
                info = None
                print(f"Error fetching Wikidata info: {e}")

            if info:
                print(f"Description: {info.get('description')}")
                print(f"Wikipedia:   {info.get('wikipedia_url')}")
                if info.get("image_url"):
                    print(f"Image:       {info.get('image_url')}")
            else:
                print("No additional info found on Wikidata.")
        else:
            print("No Wikidata ID found (top disease is not a Wikidata Q-id).")
    else:
        print("No Wikidata info available (missing final result or KG candidates).")

    # 5) Explanation (use query-based kg_candidates)
    print("-" * 30 + " Explanation " + "-" * 30)

    explainer = components.get("explainer")
    explanation = None
    if explainer and kg_candidates:
        top_disease = kg_candidates[0]
        try:
            explanation = explainer.explain(
                symptoms=text,
                disease=top_disease["disease_name"],
                confidence=float(top_disease.get("similarity_score", 0.0)),
            )
            print(f"\nExplanation for {top_disease['disease_name']}:")
            print(explanation)
        except Exception as e:
            print(f"Could not generate explanation: {e}")
    else:
        if not explainer:
            print("Explainer not available.")
        else:
            print("No KG candidates available to explain.")

    # 6) TRIAGE
    print("-" * 30 + " Triage " + "-" * 30)
    triage_result = None

    try:
        disease_iri = kg_candidates[0].get("disease_uri") if kg_candidates else None

        triage_result = triage_case(
            g=g,
            user_text=text,
            symptom_iris=symptom_iris,
            disease_iri=disease_iri,
            temperatureC=temperature,
            systolicBP=systolicBP,
            painScale=painScale,
            has_symptom_matches=bool(symptom_iris),
        )

        print("Triage final:", triage_result["final"])
        if triage_result.get("seeDoctorText"):
            print("seeDoctor:", triage_result["seeDoctorText"])
        print("Scorepoint summation:", triage_result["kg"].score_sum)

    except Exception as e:
        print(f"Error running triage engine: {e}")

    return {"kg_candidates": kg_candidates,
            "explanation": explanation,
            "triage_result": triage_result}


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main() -> None:
    stats = compute_stats()
    print("Dataset Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    components = load_components(base_dir)
    UI.start_UI(components)


if __name__ == "__main__":
    main()

