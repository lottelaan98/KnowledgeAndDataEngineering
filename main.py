import joblib

from reasoning.reasoning_engine import ReasoningEngine
from rag.rag_engine import RAGExplainer

MODEL_PATH = "models/classifier.joblib"

classifier = joblib.load(MODEL_PATH)
reasoner = ReasoningEngine()
explainer = RAGExplainer()

user_text = "i am shitting my pants and have a fever and a headache"

# PART 1 — Classifier
probs = classifier.predict_proba([user_text])[0]
labels = classifier.classes_

top_idx = probs.argmax()
prediction = {
    "disease_id": labels[top_idx],
    "score": probs[top_idx]
}

# # PART 2 — Reasoning (noop for now)
final_candidates = reasoner.rank_diseases([prediction])

# # PART 3 — Explanation
result = final_candidates[0]
explanation = explainer.explain(
    symptoms=user_text,
    disease=result["disease_id"],
    confidence=result["score"]
)

print("Prediction:", result)
print("\nExplanation:")
print(explanation)
