from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QCheckBox,
    QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QMainWindow, QScrollArea, QFrame
)
from PyQt6.QtGui import QPixmap
from pathlib import Path
import sys

# --- Logo path ---
UI_DIR = Path(__file__).parent
PATH = UI_DIR / "UULogo.png"
LOGO_PATH = str(PATH)

# --- Example disease data ---
example_data = {
    "q_id": "Q11664912",
    "disease_name": "Cervical spondylosis",
    "source_title": "Neck Injuries and Disorders",
    "source_url": "https://medlineplus.gov/neckinjuriesanddisorders.html",
    "summary": {
        "explanation_100_words_max": "Neck problems can occur in any part of your neck, including muscles, bones, joints, tendons, ligaments, or nerves. Neck pain is common and may also come from your shoulder, jaw, head, or upper arms. Muscle strain or tension often causes neck pain due to overuse, awkward sleeping positions, or exercise.",
        "symptoms": ["Pain", "Strain"],
        "treatment_options": "Treatment depends on the cause, but may include applying ice, taking pain relievers, getting physical therapy, or wearing a cervical collar. Surgery is rarely needed.",
        "see_a_doctor": {
            "recommended": True,
            "urgency": "routine",
            "guidance": "If you experience neck pain, it's recommended to see a doctor for proper diagnosis and treatment."
        }
    }
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Symptoms2Disease")
        self.setGeometry(100, 100, 1200, 800)

        # Input Section
        self.inputLabel = QLabel("Explain your illness using symptoms:")
        self.inputTextbox = QTextEdit()
        self.inputTextbox.setFixedHeight(150)
        self.inputTextbox.setPlaceholderText("E.g., neck pain, muscle strain")

        # Go Button
        self.goButton = QPushButton("Go")
        self.goButton.clicked.connect(self.on_go_pressed)

        # Options
        self.checkboxExplanation = QCheckBox("Explanation")
        self.checkboxExplanation.setCheckState(Qt.CheckState.Checked)

        self.checkboxEvaluation = QCheckBox("Evaluation")
        self.checkboxEvaluation.setCheckState(Qt.CheckState.Checked)

        self.topNCombo = QComboBox()
        self.topNCombo.addItems(["Top 1", "Top 3", "Top 5"])

        self.sourceCombo = QComboBox()
        self.sourceCombo.addItems(["Only KB", "Only LLM", "Both"])

        optionsLayout = QHBoxLayout()
        optionsLayout.addWidget(self.checkboxExplanation)
        optionsLayout.addWidget(self.checkboxEvaluation)
        optionsLayout.addWidget(self.topNCombo)
        optionsLayout.addWidget(self.sourceCombo)

        # Scrollable area for disease cards
        self.resultsArea = QScrollArea()
        self.resultsArea.setWidgetResizable(True)

        self.resultsWidget = QWidget()
        self.resultsLayout = QVBoxLayout(self.resultsWidget)
        self.resultsLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.resultsArea.setWidget(self.resultsWidget)

        # Logo
        self.logoLabel = QLabel()
        pixmap = QPixmap(LOGO_PATH)
        if pixmap.isNull():
            self.logoLabel.setText("Logo not found")
        else:
            scaled = pixmap.scaled(150, 60, Qt.AspectRatioMode.KeepAspectRatio)
            self.logoLabel.setPixmap(scaled)
        self.logoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Main Layout
        centralWidget = QWidget()
        mainLayout = QVBoxLayout(centralWidget)
        mainLayout.addWidget(self.inputLabel)
        mainLayout.addWidget(self.inputTextbox)
        mainLayout.addWidget(self.goButton)
        mainLayout.addLayout(optionsLayout)
        mainLayout.addWidget(self.resultsArea, stretch=3)
        mainLayout.addStretch()
        mainLayout.addWidget(self.logoLabel, stretch=0, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(centralWidget)

    # --- Function to create disease cards with rank ---
    def create_disease_card(self, disease_data, rank=None):
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            background-color: #f9f9f9; 
            border: 1px solid #ccc; 
            border-radius: 5px; 
            padding: 8px;
        """)
        card_layout = QVBoxLayout(card)
        
        # Rank
        if rank is not None:
            rank_label = QLabel(f"<b>#{rank}</b>")
            rank_label.setStyleSheet("font-size: 14px; color: #555;")
            card_layout.addWidget(rank_label)
        
        # Disease Name
        name_label = QLabel(f"<b>{disease_data['disease_name']}</b>")
        name_label.setStyleSheet("font-size: 16px;")
        card_layout.addWidget(name_label)

        # Source
        source_label = QLabel(f"<a href='{disease_data['source_url']}'>{disease_data['source_title']}</a>")
        source_label.setOpenExternalLinks(True)
        card_layout.addWidget(source_label)

        # Symptoms
        symptoms_label = QLabel("<b>Symptoms:</b> " + ", ".join(disease_data['summary']['symptoms']))
        card_layout.addWidget(symptoms_label)

        # Explanation
        explanation_label = QLabel("<b>Explanation:</b> " + disease_data['summary']['explanation_100_words_max'])
        explanation_label.setWordWrap(True)
        card_layout.addWidget(explanation_label)

        # Treatment
        treatment_label = QLabel("<b>Treatment:</b> " + disease_data['summary']['treatment_options'])
        treatment_label.setWordWrap(True)
        card_layout.addWidget(treatment_label)

        # See a Doctor 
        see_doctor = disease_data['summary']['see_a_doctor']
        doctor_text = f"<b>See a Doctor:</b> Recommended: {'Yes' if see_doctor['recommended'] else 'No'}, " \
                      f"Urgency: {see_doctor['urgency']}, Guidance: {see_doctor['guidance']}"
        doctor_label = QLabel(doctor_text)
        doctor_label.setWordWrap(True)
        card_layout.addWidget(doctor_label)

        return card

    def on_go_pressed(self):
        user_input = self.inputTextbox.toPlainText().strip()
        if not user_input:
            print("No explanation entered!")
            return

        source_choice = self.sourceCombo.currentText()
        top_n = self.topNCombo.currentText()
        n = int(top_n.split()[1])

        # Just for testing= show sample data
        results = []
        for i in range(n):
            results.append(example_data)  

        for i in reversed(range(self.resultsLayout.count())):
            widget = self.resultsLayout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for i, disease in enumerate(results):
            card = self.create_disease_card(disease, rank=i+1)
            self.resultsLayout.addWidget(card)

    def query_kb(self, text):
        print(f"Querying KB with: {text}")
        return example_data

    def query_llm(self, text):
        print(f"Querying LLM with: {text}")
        return example_data

    def query_both(self, text):
        print(f"Querying KB and LLM with: {text}")
        return example_data

app = QApplication(sys.argv)
window = MainWindow()
window.showMaximized()
app.exec()