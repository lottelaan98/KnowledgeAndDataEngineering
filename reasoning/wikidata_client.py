from typing import Optional, Dict
import requests
import time

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

class WikidataClient:
    """
    Client for querying the Wikidata SPARQL endpoint.
    Retrieves live description, images, and links for diseases.
    """
    
    def __init__(self, user_agent: str = "DiseasePredictor/1.0 (mailto:your-email@example.com)"):
        # Wikidata requires a User-Agent header
        self.headers = {"User-Agent": user_agent}

    def fetch_disease_info(self, wikidata_id: str) -> Optional[Dict[str, str]]:
        """
        Fetch description, image, and Wikipedia URL for a given Wikidata entity ID (e.g., Q30953).
        """
        if not wikidata_id or not wikidata_id.startswith("Q"):
            return None

        query = f"""
        SELECT ?description ?image ?article WHERE {{
            wd:{wikidata_id} schema:description ?description .
            OPTIONAL {{ wd:{wikidata_id} wdt:P18 ?image . }}
            OPTIONAL {{
                ?article schema:about wd:{wikidata_id} .
                ?article schema:isPartOf <https://en.wikipedia.org/> .
            }}
            FILTER(LANG(?description) = "en")
        }}
        LIMIT 1
        """

        try:
            response = requests.get(
                WIKIDATA_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", {}).get("bindings", [])
            if not results:
                return None

            result = results[0]
            info = {
                "description": result.get("description", {}).get("value"),
                "image_url": result.get("image", {}).get("value"),
                "wikipedia_url": result.get("article", {}).get("value")
            }
            return info

        except Exception as e:
            # Silently fail for online queries to avoid crashing the main app
            print(f"Warning: Wikidata query failed: {e}")
            return None
