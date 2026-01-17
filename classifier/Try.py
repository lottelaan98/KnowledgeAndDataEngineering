import joblib
import numpy as np

MODEL_PATH = r"C:\Users\lotte\Documents\GitHub\KnowledgeAndDataEngineering\models\classifier.joblib"

pipeline = joblib.load(MODEL_PATH)
vectorizer = pipeline.named_steps["tfidf"]
clf = pipeline.named_steps["clf"]

feature_names = vectorizer.get_feature_names_out()
classes = clf.classes_

TOP_N = 10

for i, disease in enumerate(classes):
    coefs = clf.coef_[i]
    top_idx = np.argsort(coefs)[-TOP_N:][::-1]  # highest weights

    print("\n" + "=" * 60)
    print(f"Disease: {disease} | Top {TOP_N} matched symptoms/features")
    for j in top_idx:
        print(f"{feature_names[j]:<35} weight={coefs[j]:.4f}")

