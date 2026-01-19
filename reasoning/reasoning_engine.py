from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
import importlib.util

from rdflib import Graph


class ReasoningEngine:
    """
    Hybrid Reasoning Engine (backwards-compatible output).

    - If rdf_candidates is provided: behaves like the original ReasoningEngine.
    - Else: tries query pipeline (symptom_matcher + scriptV3).
      If that fails: falls back to rdf_disease_finder.py.

    IMPORTANT: Output format and reasoning strings match the original engine.
    """

    # -----------------------------
    # Public API
    # -----------------------------
    def fuse_results(
        self,
        ml_prediction: Dict[str, Any],
        rdf_candidates: Optional[List[Dict[str, Any]]],
        user_symptoms: List[str],
        rdf_finder=None,
        ttl_path: Optional[str] = None,
        exclude_label: Optional[str] = None,
        top_k: int = 5,
        symptom_match_threshold: float = 0.10,
    ) -> Dict[str, Any]:

        # -----------------------------
        # Make output EXACTLY like original
        # -----------------------------
        disease_id = ml_prediction.get("disease_id")
        ml_score = float(ml_prediction.get("score", 0.0))

        final_result: Dict[str, Any] = {
            "disease": disease_id,
            "original_score": ml_score,
            "final_score": ml_score,
            "reasoning": [],
        }

        # -----------------------------
        # If candidates not provided, compute them (query -> fallback)
        # -----------------------------
        candidates = rdf_candidates or []
        if not candidates:
            candidates = self._get_kg_candidates_safely(
                user_symptoms=user_symptoms,
                ttl_path=ttl_path,
                top_k=top_k,
                exclude_label=exclude_label,
                symptom_match_threshold=symptom_match_threshold,
            )

        # normalize candidates so old logic works
        candidates = self._normalize_candidates(candidates)

        # -----------------------------
        # 1. Agreement Check (Top 3)
        # -----------------------------
        top3 = candidates[:3]
        kg_agrees = any(self._same(d.get("disease_name"), disease_id) for d in top3)

        if kg_agrees:
            final_result["final_score"] = min(1.0, final_result["final_score"] + 0.2)
            final_result["reasoning"].append("Knowledge Graph agrees (Bonus +20%)")
        else:
            final_result["reasoning"].append("Knowledge Graph suggests different diseases")

        # -----------------------------
        # 2. Sanity Check (Primary Symptoms)
        # -----------------------------
        if rdf_finder is not None and hasattr(rdf_finder, "get_primary_symptoms"):
            primary_symptoms = rdf_finder.get_primary_symptoms(disease_id) or []

            if primary_symptoms:
                user_sym_norm = [str(s).lower() for s in user_symptoms]
                prim_sym_norm = [str(s).lower() for s in primary_symptoms]

                has_primary = any(ps in user_sym_norm for ps in prim_sym_norm)
                if not has_primary:
                    final_result["final_score"] *= 0.5
                    final_result["reasoning"].append(
                        f"Missing primary symptoms for {disease_id} "
                        f"(Expected: {', '.join(primary_symptoms)}) (Penalty -50%)"
                    )
                else:
                    final_result["reasoning"].append("User has primary symptoms")

        # -----------------------------
        # 3. Fallback Logic
        # -----------------------------
        if final_result["final_score"] < 0.4 and candidates:
            top_rdf = candidates[0]
            final_result["reasoning"].append(
                f"ML confidence too low ({final_result['final_score']:.2%}). "
                f"Falling back to top Knowledge Graph result."
            )
            final_result["disease"] = top_rdf.get("disease_name")
            final_result["final_score"] = float(top_rdf.get("similarity_score", 0.0))
            final_result["is_fallback"] = True
        else:
            final_result["is_fallback"] = False

        return final_result

    def rank_diseases(self, disease_candidates):
        return disease_candidates

    # -----------------------------
    # Internals
    # -----------------------------
    @staticmethod
    def _same(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        return str(a).strip().lower() == str(b).strip().lower()

    @staticmethod
    def _normalize_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure candidates have keys expected by the original ReasoningEngine logic:
          - disease_name
          - similarity_score
        """
        out: List[Dict[str, Any]] = []
        for c in candidates or []:
            cc = dict(c)

            # query pipeline uses: label/finalScore
            if "disease_name" not in cc and "label" in cc:
                cc["disease_name"] = cc["label"]

            if "similarity_score" not in cc:
                if "finalScore" in cc and cc["finalScore"] is not None:
                    cc["similarity_score"] = float(cc["finalScore"])
                else:
                    cc["similarity_score"] = float(cc.get("similarity_score", 0.0) or 0.0)

            out.append(cc)

        out.sort(key=lambda r: float(r.get("similarity_score", 0.0)), reverse=True)
        return out

    # -----------------------------
    # Project root / module paths
    # -----------------------------
    @staticmethod
    def _find_project_root(start: Path) -> Path:
        """
        Walk upward until we find a folder that looks like your repo root:
        it contains ontology/, reasoning/, querying/ (at least ontology + reasoning).
        """
        cur = start.resolve()
        for _ in range(12):
            if (cur / "ontology").exists() and (cur / "reasoning").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        # fallback: directory containing this file's parent
        return start.resolve().parent

    def _project_paths(self) -> Dict[str, Path]:
        """
        Returns absolute Paths for common modules inside the repo.
        """
        # This file is usually .../reasoning/reasoning_engine3.py
        here = Path(__file__).resolve()
        root = self._find_project_root(here.parent)

        paths = {
            "root": root,
            "symptom_matcher": root / "reasoning" / "symptom_matcher.py",
            "scriptV3": root / "querying" / "scriptV3.py",
            "rdf_disease_finder": root / "reasoning" / "rdf_disease_finder.py",
        }
        return paths

    # -----------------------------
    # KG retrieval: query pipeline -> fallback
    # -----------------------------
    def _get_kg_candidates_safely(
        self,
        user_symptoms: List[str],
        ttl_path: Optional[str],
        top_k: int,
        exclude_label: Optional[str],
        symptom_match_threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Try query pipeline first; if it fails, fallback to rdf_disease_finder.py.
        Returns candidates (possibly empty).
        """
        if not ttl_path:
            return []

        try:
            return self._candidates_from_query_pipeline(
                ttl_path=ttl_path,
                user_symptoms=user_symptoms,
                top_k=top_k,
                exclude_label=exclude_label,
                threshold=symptom_match_threshold,
            )
        except Exception:
            return self._fallback_candidates_from_rdf_disease_finder(
                user_symptoms=user_symptoms,
                ttl_path=ttl_path,
                top_k=top_k,
            )

    def _candidates_from_query_pipeline(
        self,
        ttl_path: str,
        user_symptoms: List[str],
        top_k: int,
        exclude_label: Optional[str],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        symptom_matcher -> scriptV3 Query1 (+ Query3a)
        Returns candidates in dict format.
        """
        paths = self._project_paths()

        sm = self._import_module_from_path("symptom_matcher_mod", str(paths["symptom_matcher"]))
        scriptV3 = self._import_module_from_path("scriptV3_mod", str(paths["scriptV3"]))

        g = Graph()
        # IMPORTANT: parse as local file path, not URI
        g.parse(source=Path(ttl_path).resolve().as_posix(), format="turtle")

        text = " ".join(user_symptoms) if user_symptoms else ""
        matches = sm.match_symptoms(g, text, top_k=25)
        good = sm.symptoms_above_threshold(matches, threshold=threshold)
        if not good:
            raise RuntimeError("No symptom matches above threshold")

        wd_symps: List[str] = []
        for m in good:
            wd = self._uri_to_wd_prefixed(str(m["uri"]))
            if wd:
                wd_symps.append(wd)

        if not wd_symps:
            raise RuntimeError("Could not map matched symptom URIs to wd:Q IDs")

        q1_rows = scriptV3.query1_topk_diseases_by_score(
            g=g,
            symps_list=wd_symps,
            top_k_results=top_k,
            exclude_label=exclude_label,
        )

        out: List[Dict[str, Any]] = []
        for row in q1_rows or []:
            disease_iri = row.get("disease_uri")
            disease_name = row.get("label")
            score = float(row.get("finalScore", 0.0) or 0.0)

            matched_labels = scriptV3.query3_matching_symptoms(
                g=g,
                disease_iri=disease_iri,
                symps_list=wd_symps,
            )

            out.append(
                {
                    "disease_name": disease_name,
                    "similarity_score": score,
                    "matched_symptoms": matched_labels,
                    "disease_uri": disease_iri,
                }
            )

        return out

    @staticmethod
    def _uri_to_wd_prefixed(uri: str) -> Optional[str]:
        if "wikidata.org/entity/Q" in uri:
            qid = uri.rsplit("/", 1)[-1]
            if qid.startswith("Q"):
                return f"wd:{qid}"
        return None

    def _fallback_candidates_from_rdf_disease_finder(
        self,
        user_symptoms: List[str],
        ttl_path: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Fallback: call rdf_disease_finder.py (pure Python finder).
        """
        try:
            paths = self._project_paths()
            mod = self._import_module_from_path("rdf_disease_finder_mod", str(paths["rdf_disease_finder"]))
            FinderCls = getattr(mod, "RDFDiseaseFinder", None)
            if FinderCls is None:
                return []

            finder = FinderCls(str(ttl_path))
            results = finder.find_nearest_diseases(user_symptoms, top_k=top_k)

            out: List[Dict[str, Any]] = []
            for r in results or []:
                out.append(
                    {
                        "disease_name": r.get("disease_name"),
                        "similarity_score": float(r.get("similarity_score", 0.0)),
                        "matched_symptoms": r.get("matched_symptoms", []),
                        "disease_uri": str(r.get("disease_uri")) if r.get("disease_uri") is not None else None,
                    }
                )
            return out
        except Exception:
            return []

    # -----------------------------
    # Dynamic import helper
    # -----------------------------
    @staticmethod
    def _import_module_from_path(module_name: str, file_path: str):
        p = Path(file_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Module file not found: {p}")

        spec = importlib.util.spec_from_file_location(module_name, str(p))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot create spec for: {p}")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod


def main() -> None:
    """
    Minimal end-to-end demo for reasoning_engine3.py.
    Works without hardcoded absolute paths.
    """
    here = Path(__file__).resolve()
    root = ReasoningEngine._find_project_root(here.parent)
    ttl_path = root / "ontology" / "databaseV7.ttl"

    ml_prediction = {"disease_id": "Malaria", "score": 0.52}
    user_symptoms = ["fever", "headache", "fatigue"]

    rdf_finder = None
    try:
        # Optional sanity-check provider (only if module exists)
        from reasoning.rdf_disease_finder import RDFDiseaseFinder  # type: ignore
        if ttl_path.exists():
            rdf_finder = RDFDiseaseFinder(str(ttl_path))
    except Exception:
        rdf_finder = None

    engine = ReasoningEngine()

    final = engine.fuse_results(
        ml_prediction=ml_prediction,
        rdf_candidates=None,
        user_symptoms=user_symptoms,
        rdf_finder=rdf_finder,
        ttl_path=str(ttl_path),
        exclude_label=None,
        top_k=5,
        symptom_match_threshold=0.10,
    )

    print("\n=== TOP-5 KG CANDIDATES ===")
    cands = engine._get_kg_candidates_safely(
        user_symptoms=user_symptoms,
        ttl_path=str(ttl_path),
        top_k=5,
        exclude_label=None,
        symptom_match_threshold=0.10,
    )
    cands = engine._normalize_candidates(cands)
    if not cands:
        print("(no KG candidates found)")
    else:
        for i, c in enumerate(cands[:5], 1):
            print(f"{i}. {c.get('disease_name')} ({float(c.get('similarity_score', 0.0)):.3f})")

    print("\n=== FINAL REASONED PREDICTION ===")
    print("Disease:       ", final.get("disease"))
    print("Original score:", f"{float(final.get('original_score', 0.0)):.2f}")
    print("Final score:   ", f"{float(final.get('final_score', 0.0)):.2f}")
    print("Fallback used: ", final.get("is_fallback"))

    print("\n--- Reasoning steps ---")
    for step in final.get("reasoning", []):
        try:
            print("•", step)
        except UnicodeEncodeError:
            print("•", step.encode("ascii", "ignore").decode())


if __name__ == "__main__":
    main()
