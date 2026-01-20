from __future__ import annotations

from typing import List, Dict, Any, Optional


class ReasoningEngine:
    """
    Pure fuser: does NOT compute symptom matches or KG disease candidates.

    Inputs:
      - ml_prediction: {"disease_id": str, "score": float}
      - kg_candidates: list of diseases from Query pipeline (already ranked)
      - symptom_matches: list of symptoms from symptom_matcher (uri/label/score)

    Output (backwards-compatible with your original engine):
      {
        "disease": str,
        "original_score": float,
        "final_score": float,
        "reasoning": [str, ...],
        "is_fallback": bool
      }

    Extra fields (safe to ignore by old callers):
      - "symptom_matches"
      - "kg_candidates"
    """

    def fuse_results(
        self,
        ml_prediction: Dict[str, Any],
        kg_candidates: Optional[List[Dict[str, Any]]] = None,
        symptom_matches: Optional[List[Dict[str, Any]]] = None,
        rdf_finder=None,
        low_conf_threshold: float = 0.40,
        agreement_bonus: float = 0.20,
        primary_penalty: float = 0.50,
    ) -> Dict[str, Any]:

        disease_id = ml_prediction.get("disease_id")
        ml_score = float(ml_prediction.get("score", 0.0))

        final_result: Dict[str, Any] = {
            "disease": disease_id,
            "original_score": ml_score,
            "final_score": ml_score,
            "reasoning": [],
            "is_fallback": False,
        }

        # Keep upstream outputs available for debugging (backward-safe)
        final_result["symptom_matches"] = symptom_matches or []

        candidates = self._normalize_kg_candidates(kg_candidates or [])
        final_result["kg_candidates"] = candidates

        # If no KG candidates provided, ML-only result
        if not candidates:
            final_result["reasoning"].append("No Knowledge Graph candidates provided (ML only)")
            return final_result

        # -----------------------------
        # 1) Agreement check (Top 3)
        # -----------------------------
        top3 = candidates[:3]
        kg_agrees = any(self._same(c.get("disease_name"), disease_id) for c in top3)

        if kg_agrees:
            final_result["final_score"] = min(1.0, final_result["final_score"] + agreement_bonus)
            final_result["reasoning"].append("Knowledge Graph agrees (Bonus +20%)")
        else:
            final_result["reasoning"].append("Knowledge Graph suggests different diseases")

        # -----------------------------
        # 2) Sanity check (Primary symptoms) - optional
        # -----------------------------
        # We use symptom_matcher labels as "user symptoms" here (more reliable than raw text).
        user_symptom_labels = [
            str(m.get("label", "")).strip()
            for m in (symptom_matches or [])
            if m.get("label")
        ]

        if rdf_finder is not None and hasattr(rdf_finder, "get_primary_symptoms") and disease_id:
            try:
                primary_symptoms = rdf_finder.get_primary_symptoms(disease_id) or []
            except Exception:
                primary_symptoms = []

            if primary_symptoms:
                user_sym_norm = [s.lower() for s in user_symptom_labels]
                prim_sym_norm = [str(s).lower() for s in primary_symptoms]

                has_primary = any(ps in user_sym_norm for ps in prim_sym_norm)
                if not has_primary:
                    final_result["final_score"] *= primary_penalty
                    final_result["reasoning"].append(
                        f"Missing primary symptoms for {disease_id} "
                        f"(Expected: {', '.join(primary_symptoms)}) (Penalty -50%)"
                    )
                else:
                    final_result["reasoning"].append("User has primary symptoms")

        # -----------------------------
        # 3) Fallback logic (low ML confidence -> use KG top result)
        # -----------------------------
        if final_result["final_score"] < low_conf_threshold:
            top_kg = candidates[0]
            final_result["reasoning"].append(
                f"ML confidence too low ({final_result['final_score']:.2%}). "
                f"Falling back to top Knowledge Graph result."
            )
            final_result["disease"] = top_kg.get("disease_name")
            final_result["final_score"] = float(top_kg.get("similarity_score", 0.0))
            final_result["is_fallback"] = True

        return final_result

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _same(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        return str(a).strip().lower() == str(b).strip().lower()

    @staticmethod
    def _normalize_kg_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Accept KG candidates in either "percent" or "0..1" score form.

        Supported input keys per candidate:
          - disease_name or label
          - disease_uri or disease_iri or disease_uri
          - similarity_pct (e.g. 70.55) OR similarity_score (0..1) OR finalScore (0..1)
          - matched_symptoms (list of strings) OR matched (list of strings)

        Output candidates always contain:
          - disease_name
          - disease_uri
          - similarity_score (0..1 float)
          - similarity_pct (0..100 float)
          - matched_symptoms (list[str])
        """
        out: List[Dict[str, Any]] = []

        for c in cands or []:
            cc = dict(c)

            # disease name normalization
            if "disease_name" not in cc:
                if "label" in cc:
                    cc["disease_name"] = cc["label"]

            # disease uri normalization
            if "disease_uri" not in cc:
                if "disease_iri" in cc:
                    cc["disease_uri"] = cc["disease_iri"]
                elif "disease_uri" in cc:
                    pass
                else:
                    cc["disease_uri"] = None

            # matched symptoms normalization
            if "matched_symptoms" not in cc or cc["matched_symptoms"] is None:
                if "matched" in cc and cc["matched"] is not None:
                    cc["matched_symptoms"] = cc["matched"]
                else:
                    cc["matched_symptoms"] = []

            # similarity normalization (always make both score + pct)
            sim_score: float = 0.0
            sim_pct: float = 0.0

            if cc.get("similarity_pct") is not None:
                sim_pct = float(cc["similarity_pct"])
                sim_score = sim_pct / 100.0
            elif cc.get("similarity_score") is not None:
                sim_score = float(cc["similarity_score"])
                sim_pct = sim_score * 100.0
            elif cc.get("finalScore") is not None:
                sim_score = float(cc["finalScore"])
                sim_pct = sim_score * 100.0

            cc["similarity_score"] = sim_score
            cc["similarity_pct"] = sim_pct

            out.append(cc)

        out.sort(key=lambda r: float(r.get("similarity_score", 0.0)), reverse=True)
        return out