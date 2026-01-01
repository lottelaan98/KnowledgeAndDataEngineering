class ReasoningEngine:
    """
    This class must NEVER see raw text.
    It operates ONLY on canonical IDs.
    """

    def rank_diseases(self, disease_candidates):
        """
        Input:
        [
          {"disease_id": "DIS_004", "score": 0.81}
        ]

        Output:
        Same structure, possibly re-ranked or filtered.
        """

        # 🔌 PLACEHOLDER
        # Later:
        # - Query Neo4j
        # - Apply rules
        # - Fuse KG score with ML score

        return disease_candidates
