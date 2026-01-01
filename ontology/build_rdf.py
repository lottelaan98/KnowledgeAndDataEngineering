import pandas as pd
import re
from rdflib import Graph, Namespace, Literal, RDF

# ---------- CONFIG ----------
CSV_PATH = "data/symptom2disease.csv"
OUTPUT_TTL = "ontology/knowledge.ttl"

BASE_URI = "http://uu.nl/medical/"
EX = Namespace(BASE_URI)

# Very simple symptom extractor (v1 – replace later)
KNOWN_SYMPTOMS = [
    "fever", "headache", "nausea", "vomiting", "cough",
    "fatigue", "diarrhea", "pain", "sore throat"
]
# ----------------------------


def normalize(text):
    return re.sub(r"[^a-zA-Z ]", "", text.lower())


def extract_symptoms(text):
    text = normalize(text)
    return [s for s in KNOWN_SYMPTOMS if s in text]


def uriify(name):
    return name.replace(" ", "").replace("-", "")


def main():
    df = pd.read_csv(CSV_PATH)

    g = Graph()
    g.bind("ex", EX)

    for _, row in df.iterrows():
        disease_name = row["label"].strip()
        text = row["text"]

        disease_uri = EX[uriify(disease_name)]
        g.add((disease_uri, RDF.type, EX.Disease))

        symptoms = extract_symptoms(text)

        for symptom in symptoms:
            symptom_uri = EX[uriify(symptom)]
            g.add((symptom_uri, RDF.type, EX.Symptom))
            g.add((disease_uri, EX.hasSymptom, symptom_uri))

        if symptoms:
            explanation = (
                f"{disease_name} is commonly associated with "
                + ", ".join(symptoms)
                + "."
            )
            g.add((
                disease_uri,
                EX.hasExplanation,
                Literal(explanation)
            ))

    g.serialize(destination=OUTPUT_TTL, format="turtle")
    print(f"RDF knowledge graph written to {OUTPUT_TTL}")


if __name__ == "__main__":
    main()
