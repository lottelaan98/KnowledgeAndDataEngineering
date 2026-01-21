from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QCheckBox,
    QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QMainWindow, QScrollArea, QFrame, QDoubleSpinBox, QSpinBox, QGroupBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtGui import QPixmap
from pathlib import Path
import sys
import json
from typing import List, Dict, Any, Optional
from main import run_diagnosis

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
        self.setGeometry(100, 100, 1280, 900)

        # Fonts
        self.titleFont = QFont("Arial", 11, QFont.Weight.Bold)
        self.subtitleFont = QFont("Arial", 10, QFont.Weight.DemiBold)
        self.normalFont = QFont("Arial", 10)

        # Input
        self.inputLabel = QLabel("<b>Describe your symptoms:</b>")
        self.inputLabel.setFont(self.subtitleFont)
        self.inputTextbox = QTextEdit()
        self.inputTextbox.setFixedHeight(150)
        self.inputTextbox.setFont(self.normalFont)
        self.inputTextbox.setPlaceholderText("E.g., I have a fever, headache, and nausea...")

        # Optional Specifications
        self.tempCheck = QCheckBox("Temperature (°C)")
        self.tempCheck.stateChanged.connect(self.toggle_temp)
        self.tempSpin = QDoubleSpinBox()
        self.tempSpin.setRange(30.0, 45.0)
        self.tempSpin.setSingleStep(0.1)
        self.tempSpin.setValue(37.0)
        self.tempSpin.setEnabled(False)
        self.tempSpin.setToolTip("Set your body temperature in Celsius.")

        self.bpCheck = QCheckBox("Systolic Blood Pressure (mmHg)")
        self.bpCheck.stateChanged.connect(self.toggle_bp)
        self.bpSpin = QSpinBox()
        self.bpSpin.setRange(70, 250)
        self.bpSpin.setValue(120)
        self.bpSpin.setEnabled(False)
        self.bpSpin.setToolTip("Set your systolic blood pressure in mmHg.")

        self.painCheck = QCheckBox("Pain Scale (0-10)")
        self.painCheck.stateChanged.connect(self.toggle_pain)
        self.painSpin = QSpinBox()
        self.painSpin.setRange(0, 10)
        self.painSpin.setValue(5)
        self.painSpin.setEnabled(False)
        self.painSpin.setToolTip("Set your pain scale from 0 (no pain) to 10 (worst pain).")

        specLayout = QHBoxLayout()
        specLayout.setSpacing(20)
        specLayout.addWidget(self.tempCheck)
        specLayout.addWidget(self.tempSpin)
        specLayout.addWidget(self.bpCheck)
        specLayout.addWidget(self.bpSpin)
        specLayout.addWidget(self.painCheck)
        specLayout.addWidget(self.painSpin)

        self.specifications = QGroupBox("Optional Specifications")
        self.specifications.setFont(self.subtitleFont)
        self.specifications.setLayout(specLayout)
        self.specifications.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                margin-top: 20px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)

        # Options
        self.checkboxExplanation = QCheckBox("Explanation of Diagnosis")
        self.checkboxExplanation.setToolTip( "Provides an explanation of the diagnosis using retrieved medical knowledge.")
        self.checkboxEvaluation = QCheckBox("Show Confidence Level")
        self.checkboxEvaluation.setToolTip( "Includes advanced statistics used for more precise diagnosis" )
        self.topNCombo = QComboBox()
        self.topNCombo.addItems(["Top 1 Diseases", "Top 3 Diseases", "Top 5 Diseases"])
        self.topNCombo.setCurrentIndex(2)
        self.topNCombo.setToolTip( "Controls how many diagnoses are returned.\n" "Top 1: Most likely diagnosis only\n" "Top 3: Three most likely diagnoses\n" "Top 5: Broader differential diagnosis" )

        optionsLayout = QHBoxLayout()
        optionsLayout.setSpacing(20)
        optionsLayout.addWidget(self.checkboxExplanation)
        optionsLayout.addWidget(self.checkboxEvaluation)
        optionsLayout.addWidget(QLabel("Number of suggestions:"))
        optionsLayout.addWidget(self.topNCombo)

        self.options = QGroupBox("Options")
        self.options.setFont(self.subtitleFont)
        self.options.setLayout(optionsLayout)
        self.options.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                margin-top: 20px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)

        # Go Button
        self.goButton = QPushButton("Diagnose")
        self.goButton.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                background-color: #1976D2;
                color: white;
                padding: 8px 20px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        self.goButton.clicked.connect(self.on_go_pressed)

        # Results Area
        self.resultsArea = QScrollArea()
        self.resultsArea.setWidgetResizable(True)
        self.resultsWidget = QWidget()
        self.resultsLayout = QVBoxLayout(self.resultsWidget)
        self.resultsLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.resultsLayout.setSpacing(12)
        self.resultsArea.setWidget(self.resultsWidget)

        # Disclaimer
        self.disclaimerLabel = QLabel(
            "This tool is for informational purposes only and does not constitute medical advice."
        )
        self.disclaimerLabel.setWordWrap(True)
        self.disclaimerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.disclaimerLabel.setStyleSheet("color: #b00020; font-size: 11px;")

        # Logo
        self.logoLabel = QLabel()
        pixmap = QPixmap(LOGO_PATH)
        if not pixmap.isNull():
            self.logoLabel.setPixmap(pixmap.scaled(150, 60, Qt.AspectRatioMode.KeepAspectRatio))
        self.logoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Main Layout
        centralWidget = QWidget()
        mainLayout = QVBoxLayout(centralWidget)
        mainLayout.setContentsMargins(15, 15, 15, 15)
        mainLayout.setSpacing(15)
        mainLayout.addWidget(self.inputLabel)
        mainLayout.addWidget(self.inputTextbox)
        mainLayout.addWidget(self.specifications)
        mainLayout.addWidget(self.options)
        mainLayout.addWidget(self.goButton, alignment=Qt.AlignmentFlag.AlignCenter)
        mainLayout.addWidget(self.resultsArea, stretch=3)
        mainLayout.addWidget(self.disclaimerLabel)
        mainLayout.addWidget(self.logoLabel)
        self.setCentralWidget(centralWidget)

    def on_go_pressed(self):
        text = self.inputTextbox.toPlainText().strip()
        if not text:
            return
        
        raw_results = run_diagnosis(
            text=text,
            components=self.components,
            temperature=self.tempSpin.value() if self.tempCheck.isChecked() else None,
            systolicBP=self.bpSpin.value() if self.bpCheck.isChecked() else None,
            painScale=self.painSpin.value() if self.painCheck.isChecked() else None,
        )

        # Clear previous results
        while self.resultsLayout.count():
            self.resultsLayout.takeAt(0).widget().deleteLater()

        top_n = int(self.topNCombo.currentText().split()[1])

        kg_candidates = raw_results.get("kg_candidates", [])

        if not kg_candidates:
            self.resultsLayout.addWidget(QLabel(
                "No diseases found. Try different symptoms or wording."
            ))
            return

        # Enrich each KG candidate
        cards = []
        for r in kg_candidates:
            disease_uri = r["disease_uri"]
            similarity = r.get("similarity_pct", 0.0)

            card_data = self.build_card_from_uri(
                disease_uri=disease_uri,
                similarity_pct=similarity,
            )
            if card_data:
                cards.append(card_data)

        # Render top-N
        for i, card in enumerate(cards[:top_n], 1):
            self.resultsLayout.addWidget(self.create_disease_card(card, i, raw_results.get("explanation"), raw_results.get("triage_result")))

    def build_card_from_uri(
        self,
        disease_uri: str,
        similarity_pct: float,
    ) -> Optional[Dict[str, Any]]:

        qid = extract_qid(disease_uri)
        if not qid or qid not in DISEASE_SUMMARIES:
            return None

        s = DISEASE_SUMMARIES[qid]

        return {
            "disease_name": s["disease_name"],
            "symptoms": s["summary"]["symptoms"],
            "description": s["summary"]["explanation_100_words_max"],
            "treatment": s["summary"]["treatment_options"],
            "url": s["source_url"],
            "similarity_pct": similarity_pct,
        }

    def create_disease_card(self, disease: Dict[str, Any], rank: int, explanation=None, triage=None):
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #fdfdfd;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 12px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setSpacing(6)
        layout.addWidget(QLabel(f"<b>#{rank} {disease['disease_name']}</b>"))
        layout.addWidget(QLabel("<b>Symptoms:</b> " + ", ".join(disease["symptoms"])))
        desc = QLabel("<b>Description:</b> " + disease["description"])
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addWidget(QLabel("<b>Treatment:</b> " + disease["treatment"]))

        if self.checkboxEvaluation.isChecked():
            layout.addWidget(QLabel(f"<b>Confidence:</b> {disease['similarity_pct']:.2f}%"))

        if self.checkboxExplanation.isChecked() and rank == 1 and explanation:
            expl = QLabel("<b>Explanation:</b> " + explanation)
            expl.setWordWrap(True)
            layout.addWidget(expl)

        if rank == 1 and triage:
            triSeverity = QLabel("<b>Triage Severity:</b> " + triage["final"])
            triSeverity.setWordWrap(True)
            layout.addWidget(triSeverity)
            seeDoc = QLabel("<b>Triage Advice:</b> " + triage["seeDoctorText"])
            seeDoc.setWordWrap(True)
            layout.addWidget(seeDoc)
            
        link = QLabel(f'<b>More info:</b> <a href="{disease["url"]}">{disease["url"]}</a>')
        link.setOpenExternalLinks(True) 
        link.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse) 
        layout.addWidget(link)

        return card

    def toggle_temp(self, state):
        self.tempSpin.setEnabled(state == 2)

    def toggle_bp(self, state):
        self.bpSpin.setEnabled(state == 2)

    def toggle_pain(self, state):
        self.painSpin.setEnabled(state == 2)
  


def start_UI(components):
    app = QApplication(sys.argv)
    window = MainWindow(components)
    window.showMaximized()
    app.exec()
