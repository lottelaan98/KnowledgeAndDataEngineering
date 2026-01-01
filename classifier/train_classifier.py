import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA_PATH = "data/symptom2disease.csv"
MODEL_PATH = "models/classifier.joblib"

df = pd.read_csv(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        multi_class="auto"
    ))
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))

joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
