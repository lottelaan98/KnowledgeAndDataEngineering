import os
from urllib import response
import faiss
import requests
from sentence_transformers import SentenceTransformer


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"


class RAGExplainer:
    """
    Explains predictions using retrieved documents.
    NEVER predicts diseases.
    """

    def __init__(self, docs_path="rag/docs"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = []
        self._load_docs(docs_path)
        self._build_index()

    def _load_docs(self, path):
        for fname in os.listdir(path):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                self.docs.append(f.read())

    def _build_index(self):
        embeddings = self.embedder.encode(self.docs)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def _retrieve(self, query, k=3):
        q_emb = self.embedder.encode([query])
        _, idxs = self.index.search(q_emb, k)       
        return [self.docs[i] for i in idxs[0]]

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3",
            "messages": [
                {"role": "system", "content": "You are a medical explanation assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=60
    )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def explain(self, symptoms, disease, confidence):
        """
        symptoms: raw user text OR canonical symptom list
        disease: predicted disease name or ID
        confidence: float
        """

        retrieval_query = f"{disease} symptom explanation"
        context_docs = self._retrieve(retrieval_query)

        context = "\n\n".join(context_docs)

        prompt = f"""
You are a medical explanation assistant.

Rules:
- ONLY explain the disease provided.
- DO NOT introduce new diseases.
- DO NOT give treatment advice.
- DO NOT make a diagnosis.
- Base your answer ONLY on the context.

User symptoms:
{symptoms}

Predicted disease:
{disease} (confidence: {confidence:.2f})

Context:
{context}

Explain clearly why this disease matches the symptoms.
"""

        return self._call_ollama(prompt)
