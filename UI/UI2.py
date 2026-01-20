from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QCheckBox,
    QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QMainWindow, QScrollArea, QFrame
)
from PyQt6.QtGui import QPixmap
from pathlib import Path
import sys
import json
from typing import List, Dict, Any, Optional

UI_DIR = Path(__file__).parent
LOGO_PATH = str(UI_DIR / "UULogo.png")
SUMMARY_PATH = UI_DIR.parent / "ontology" / "disease_summaries.json"

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    DISEASE_SUMMARIES = json.load(f)

def extract_qid(disease_uri: str) -> Optional[str]:
    if disease_uri and "wikidata.org/entity/" in disease_uri:
        return disease_uri.rsplit("/", 1)[-1]
    return None


class MainWindow(QMainWindow):
    def __init__(self, components):
        super().__init__()
        self.components = components

        self.setWindowTitle("Symptoms2Disease")
        self.setGeometry(100, 100, 1200, 800)

        self.inputLabel = QLabel("Explain your illness using symptoms:")
        self.inputTextbox = QTextEdit()
        self.inputTextbox.setFixedHeight(150)

        self.goButton = QPushButton("Go")
        self.goButton.clicked.connect(self.on_go_pressed)

        self.checkboxExplanation = QCheckBox("Explanation")
        self.checkboxExplanation.setChecked(True)

        self.checkboxEvaluation = QCheckBox("Evaluation")
        self.checkboxEvaluation.setChecked(True)

        self.topNCombo = QComboBox()
        self.topNCombo.addItems(["Top 1", "Top 3", "Top 5"])

        self.sourceCombo = QComboBox()
        self.sourceCombo.addItems(["KB", "ML+KG"])
        self.sourceCombo.currentTextChanged.connect(self.on_source_changed)

        optionsLayout = QHBoxLayout()
        optionsLayout.addWidget(self.checkboxExplanation)
        optionsLayout.addWidget(self.checkboxEvaluation)
        optionsLayout.addWidget(self.topNCombo)
        optionsLayout.addWidget(self.sourceCombo)

        self.resultsArea = QScrollArea()
        self.resultsArea.setWidgetResizable(True)

        self.resultsWidget = QWidget()
        self.resultsLayout = QVBoxLayout(self.resultsWidget)
        self.resultsLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.resultsArea.setWidget(self.resultsWidget)

        self.logoLabel = QLabel()
        pixmap = QPixmap(LOGO_PATH)
        if not pixmap.isNull():
            self.logoLabel.setPixmap(
                pixmap.scaled(150, 60, Qt.AspectRatioMode.KeepAspectRatio)
            )
        self.logoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        centralWidget = QWidget()
        mainLayout = QVBoxLayout(centralWidget)
        mainLayout.addWidget(self.inputLabel)
        mainLayout.addWidget(self.inputTextbox)
        mainLayout.addWidget(self.goButton)
        mainLayout.addLayout(optionsLayout)
        mainLayout.addWidget(self.resultsArea, stretch=3)
        mainLayout.addWidget(self.logoLabel)

        self.setCentralWidget(centralWidget)

    def create_disease_card(self, disease: Dict[str, Any], rank: int):
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            background-color: #f9f9f9;
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 10px;
        """)

        layout = QVBoxLayout(card)
        layout.addWidget(QLabel(f"<b>#{rank} {disease['disease_name']}</b>"))

        layout.addWidget(QLabel("<b>Symptoms:</b> " + ", ".join(disease["symptoms"])))

        if self.checkboxExplanation.isChecked():
            desc = QLabel("<b>Description:</b> " + disease["description"])
            desc.setWordWrap(True)
            layout.addWidget(desc)

            if disease.get("rag_explanation"):
                rag = QLabel("<b>Explanation:</b> " + disease["rag_explanation"])
                rag.setWordWrap(True)
                layout.addWidget(rag)

        layout.addWidget(QLabel("<b>Treatment:</b> " + disease["treatment"]))

        if self.checkboxEvaluation.isChecked():
            layout.addWidget(QLabel(f"<b>Confidence:</b> {disease['similarity_pct']:.2f}%"))

        sd = disease["see_doctor"]
        layout.addWidget(QLabel(
            f"<b>See a doctor:</b> {'Yes' if sd['recommended'] else 'No'} "
            f"({sd['urgency']}) — {sd['guidance']}"
        ))

        if disease.get("triage_text"):
            tri = QLabel("<b>Triage:</b> " + disease["triage_text"])
            tri.setWordWrap(True)
            layout.addWidget(tri)

        if self.checkboxEvaluation.isChecked() and disease.get("decision_source"):
            if disease["decision_source"] == "ML":
                label = QLabel("<b>Decision source:</b> Machine Learning (high confidence)")
                label.setStyleSheet("color: #1a7f37;")
            else:
                label = QLabel("<b>Decision source:</b> Knowledge Graph (ML confidence too low)")
                label.setStyleSheet("color: #b45309;")
            layout.addWidget(label)

        return card

    def build_card_from_uri(
        self,
        disease_uri: str,
        similarity_pct: float,
        matched_symptoms: List[str],
        text: str,
        symptom_iris: List[str],
    ) -> Optional[Dict[str, Any]]:

        from reasoning.emergency_reasoner import triage_case

        qid = extract_qid(disease_uri)
        if not qid or qid not in DISEASE_SUMMARIES:
            return None

        s = DISEASE_SUMMARIES[qid]

        rag_text = None
        if self.checkboxExplanation.isChecked():
            explainer = self.components.get("explainer")
            if explainer:
                try:
                    rag_text = explainer.explain(
                        symptoms=text,
                        disease=s["disease_name"],
                        confidence=similarity_pct / 100,
                    )
                except Exception:
                    rag_text = None

        triage_text = None
        try:
            triage = triage_case(
                g=self.components["rdf_finder"].graph,
                user_text=text,
                symptom_iris=symptom_iris,
                disease_iri=disease_uri,
                temperatureC=None,
                systolicBP=None,
                painScale=None,
                has_symptom_matches=bool(symptom_iris),
            )
            triage_text = triage.get("final")
        except Exception:
            pass

        return {
            "disease_name": s["disease_name"],
            "symptoms": s["summary"]["symptoms"],
            "description": s["summary"]["explanation_100_words_max"],
            "treatment": s["summary"]["treatment_options"],
            "see_doctor": s["summary"]["see_a_doctor"],
            "similarity_pct": similarity_pct,
            "decision_source": None,
            "rag_explanation": rag_text,
            "triage_text": triage_text,
        }

    def on_go_pressed(self):
        text = self.inputTextbox.toPlainText().strip()
        if not text:
            return

        results = self.run_pipeline(text)

        while self.resultsLayout.count():
            self.resultsLayout.takeAt(0).widget().deleteLater()

        top_n = int(self.topNCombo.currentText().split()[1])

        for i, disease in enumerate(results[:top_n], 1):
            self.resultsLayout.addWidget(self.create_disease_card(disease, i))

    def on_source_changed(self, mode: str):
        self.topNCombo.setVisible(mode != "ML+KG")

    def run_pipeline(self, text: str) -> List[Dict[str, Any]]:
        from main import extract_symptoms_with_matcher, uris_to_prefixed
        from querying import scriptV3

        rdf_finder = self.components["rdf_finder"]
        g = rdf_finder.graph

        extracted = extract_symptoms_with_matcher(text, g)
        symptom_iris = extracted["iris"]
        symptom_prefixed = uris_to_prefixed(symptom_iris)

        kg_rows = scriptV3.query1_topk_diseases_by_score(
            g=g,
            symps_list=symptom_prefixed,
            top_k_results=5,
            exclude_label=None,
        )

        kg_cards = []
        for r in kg_rows:
            matched = scriptV3.query3_matching_symptoms(
                g=g,
                disease_iri=r["disease_uri"],
                symps_list=symptom_prefixed,
            )
            card = self.build_card_from_uri(
                r["disease_uri"],
                float(r["finalScore"]) * 100,
                matched,
                text,
                symptom_iris,
            )
            if card:
                kg_cards.append(card)

        if self.sourceCombo.currentText() == "KB":
            return kg_cards

        classifier = self.components["classifier"]
        reasoner = self.components["reasoner"]

        probs = classifier.predict_proba([text])[0]
        labels = classifier.classes_
        idx = probs.argmax()

        ml_prediction = {
            "disease_id": labels[idx],
            "score": float(probs[idx]),
        }

        final = reasoner.fuse_results(
            ml_prediction=ml_prediction,
            kg_candidates=kg_rows,
            symptom_matches=extracted["matches"],
            rdf_finder=rdf_finder,
        )

        if not final.get("is_fallback"):
            card = self.build_card_from_uri(
                final["disease_uri"],
                final["final_score"] * 100,
                [],
                text,
                symptom_iris,
            )
            if card:
                card["decision_source"] = "ML"
                return [card]

        card = kg_cards[0]
        card["decision_source"] = "KG_FALLBACK"
        return [card]


def start_UI(components):
    app = QApplication(sys.argv)
    window = MainWindow(components)
    window.showMaximized()
    app.exec()
