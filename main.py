import joblib
import sys
from pathlib import Path
from typing import List, Dict, Any


base_dir = Path(__file__).parent.resolve()
sys.path.append(str(base_dir))

from reasoning.reasoning_engine import ReasoningEngine
from reasoning.rdf_disease_finder import RDFDiseaseFinder
from rag.rag_engine import RAGExplainer


# Common symptoms loaded dynamically from the KG

def extract_symptoms_from_text(text: str, known_symptoms: List[str]) -> List[str]:
    """
    Extract symptom keywords from user text.
    Simple keyword matching.
    """
    text_lower = text.lower()
    found_symptoms = []
    
    # Check for multi-word symptom
    # Sort by length to match longest phrases first (e.g. "chest pain" before "pain")
    if not known_symptoms:
        return []

    for symptom in sorted(known_symptoms, key=len, reverse=True):
        if symptom in text_lower:
            found_symptoms.append(symptom)
            text_lower = text_lower.replace(symptom, "", 1)
    
    # common phrases
    synonyms = {
        "loose stool": "diarrhea",
        "stomach ache": "abdominal pain",
        "tummy ache": "abdominal pain", 
        "high temp": "fever",
        "throwing up": "vomiting",
        "shitting": "diarrhea"
    }
    
    for phrase, symptom in synonyms.items():
        if phrase in text_lower and symptom not in found_symptoms:
            found_symptoms.append(symptom)
            
    return list(set(found_symptoms))

from reasoning.wikidata_client import WikidataClient

def load_components(base_path: Path):
    """Load all models and data."""
    print("Loading components...", file=sys.stderr)
    
    model_path = base_path / "models" / "classifier.joblib"
    rdf_path = base_path / "ontology" / "version 2 database.ttl"
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
    
    print("\n" + "="*70)
    print("Disease Prediction System Results")
    print("="*70)
    print(f"Input: {text}")
    
    # 1. Extract Symptoms
    known_symptoms = components.get('all_symptoms', [])
    if not known_symptoms:
        print("Warning: No known symptoms loaded from Knowledge Graph. Extraction may fail.")
    
    symptoms = extract_symptoms_from_text(text, known_symptoms)
    print(f"Extracted Symptoms: {', '.join(symptoms) if symptoms else 'None found'}\n")
    
    if not symptoms:
        print("No symptoms identified. Please provide more specific details.")
        return

    # 2. RDF Search
    rdf_finder = components.get('rdf_finder')
    nearest_diseases = []
    
    print("-" * 30 + " RDF Knowledge Graph " + "-" * 30)
    if rdf_finder:
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
    else:
        print("RDF Finder not initialized.")

    # 3. Hybrid Reasoning (Fusion)
    classifier = components.get('classifier')
    reasoner = components.get('reasoner')
    print("-" * 30 + " Hybrid Reasoning " + "-" * 30)
    
    if classifier and reasoner:
        try:
            # Get raw ML prediction
            probs = classifier.predict_proba([text])[0]
            labels = classifier.classes_
            top_idx = probs.argmax()
            ml_prediction = {
                "disease_id": labels[top_idx], 
                "score": float(probs[top_idx])
            }
            
            # Fuse results
            final_result = reasoner.fuse_results(
                ml_prediction=ml_prediction,
                rdf_candidates=nearest_diseases,
                user_symptoms=symptoms,
                rdf_finder=rdf_finder
            )
            
            print(f"Final Prediction: {final_result['disease']}")
            print(f"Confidence Score: {final_result['final_score']:.2%}")
            
            if final_result['is_fallback']:
                print("Note: Result based on Knowledge Graph due to low ML confidence.")
            else:
                print(f"Base ML Score:    {final_result['original_score']:.2%}")
                
            print("\nReasoning Trace:")
            for reason in final_result['reasoning']:
                try:
                    print(f"  {reason}")
                except UnicodeEncodeError:
                     print(f"  {reason.encode('ascii', 'ignore').decode()}")
            print()
            
        except Exception as e:
            print(f"Error in reasoning engine: {e}")
    else:
        print("Classifier or Reasoner not available.")

    # 4. Live Wikidata Info
    wikidata = components.get('wikidata')
    if final_result and final_result['disease']:
        print("-" * 30 + " Live Wikidata Info " + "-" * 30)
        disease_name = final_result['disease']
        wikidata_id = rdf_finder.get_wikidata_id(disease_name)
        
        if wikidata_id and wikidata:
            print(f"Fetching data for {disease_name} ({wikidata_id})...")
            info = wikidata.fetch_disease_info(wikidata_id)
            if info:
                print(f"Description: {info.get('description')}")
                print(f"Wikipedia:   {info.get('wikipedia_url')}")
                if info.get('image_url'):
                    print(f"Image:       {info.get('image_url')}")
            else:
                print("No additional info found on Wikidata.")
        else:
            print("No Wikidata ID found in ontology.")

    # 5. Explanation
    explainer = components.get('explainer')
    if nearest_diseases and explainer:
        print("-" * 30 + " Explanation " + "-" * 30)
        top_disease = nearest_diseases[0]
        try:
            explanation = explainer.explain(
                symptoms=text,
                disease=top_disease['disease_name'],
                confidence=top_disease['similarity_score']
            )
            print(f"\nExplanation for {top_disease['disease_name']}:")
            print(explanation)
        except Exception as e:
            print(f"Could not generate explanation: {e}")

def main():
    """
    Main execution function.
    Runs a sample diagnosis flow when the script is executed directly.
    """
    components = load_components(base_dir)
    
    # Sample input 
    #  Checck  symptoms
    # sample_text = "fever and cough"
    sample_text = "I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches."
    
    run_diagnosis(sample_text, components)

if __name__ == "__main__":
    main()
