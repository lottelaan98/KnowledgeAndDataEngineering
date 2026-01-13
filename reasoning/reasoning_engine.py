from typing import List, Dict, Any

class ReasoningEngine:
    """
    Hybrid Reasoning Engine.
    Fuses outputs from the Machine Learning model (Classifier) and the Knowledge Graph (RDF Finder).
    """

    def fuse_results(
        self, 
        ml_prediction: Dict[str, Any], 
        rdf_candidates: List[Dict[str, Any]], 
        user_symptoms: List[str],
        rdf_finder=None
    ) -> Dict[str, Any]:
        """
        Combine ML and RDF results to produce a final, reasoned prediction.
        
        Logic:
        1. Start with ML score.
        2. Agreement Bonus: +20% content if ML prediction is in RDF Top 3.
        3. Sanity Check: -50% content if user lacks PRIMARY symptoms of the predicted disease.
        4. Fallback: If ML confidence is too low (< 40%), prefer RDF Top 1.
        """
        
        final_result = {
            "disease": ml_prediction['disease_id'],
            "original_score": ml_prediction['score'],
            "final_score": ml_prediction['score'],
            "reasoning": []
        }
        
        # 1. Agreement Check
        kg_agrees = any(d['disease_name'] == ml_prediction['disease_id'] for d in rdf_candidates)
        if kg_agrees:
            final_result['final_score'] = min(1.0, final_result['final_score'] + 0.2)
            final_result['reasoning'].append("Knowledge Graph agrees (Bonus +20%)")
        else:
            final_result['reasoning'].append("Knowledge Graph suggests different diseases")

        # 2. Sanity Check (Primary Symptoms)
        # Only feasible if we have the RDF finder instance to query structure
        if rdf_finder:
            primary_symptoms = rdf_finder.get_primary_symptoms(ml_prediction['disease_id'])
            # Normalize to check for intersection
            user_sym_norm = [s.lower() for s in user_symptoms]
            prim_sym_norm = [s.lower() for s in primary_symptoms]
            
            # If disease has primary symptoms defined, check if user has at least one
            if primary_symptoms:
                has_primary = any(s in user_sym_norm for s in prim_sym_norm)
                if not has_primary:
                    final_result['final_score'] *= 0.5
                    final_result['reasoning'].append(
                        f"Missing primary symptoms for {ml_prediction['disease_id']} "
                        f"(Expected: {', '.join(primary_symptoms)}) (Penalty -50%)"
                    )
                else:
                    final_result['reasoning'].append("User has primary symptoms")

        # 3. Fallback Logic
        if final_result['final_score'] < 0.4 and rdf_candidates:
            top_rdf = rdf_candidates[0]
            final_result['reasoning'].append(
                f"ML confidence too low ({final_result['final_score']:.2%}). "
                f"Falling back to top Knowledge Graph result."
            )
            final_result['disease'] = top_rdf['disease_name']
            final_result['final_score'] = top_rdf['similarity_score']
            final_result['is_fallback'] = True
        else:
            final_result['is_fallback'] = False

        return final_result

    def rank_diseases(self, disease_candidates):
        # Legacy method compatibility
        return disease_candidates
