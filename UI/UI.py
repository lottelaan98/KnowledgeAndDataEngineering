
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QLabel, QComboBox, QCheckBox, QVBoxLayout, QLineEdit, QTextEdit, QHBoxLayout
from PyQt6.QtGui import QAction, QPixmap
import sys
from pathlib import Path

UI_DIR = Path(__file__).parent / "UI"
PATH = UI_DIR / "UULogo.png"
LOGO_PATH = str(PATH)


import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Symptoms2Disease")
        #self.setFixedSize(1400,800)

        # Label
        self.label = QLabel("A Little introduction")
        font = self.label.font()
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft)  # or AlignHCenter|AlignVCenter

        # Checkbox
        self.widgetBox1 = QCheckBox("Explanation")
        self.widgetBox1.setCheckState(Qt.CheckState.Checked)
        self.widgetBox1.stateChanged.connect(self.show_state)

        self.widgetBox2 = QCheckBox("Evaluation")
        self.widgetBox2.setCheckState(Qt.CheckState.Checked)
        self.widgetBox2.stateChanged.connect(self.show_state)


        # QComboBox
        self.widgetComboBox1 = QComboBox()
        self.widgetComboBox1.addItems(["Top 1", "Top 3", "Top 5"])
        # Sends the current index (position) of the selected item.
        self.widgetComboBox1.currentIndexChanged.connect( self.index_changed )
        # There is an alternate signal to send the text.
        self.widgetComboBox1.currentTextChanged.connect( self.text_changed )

        self.widgetComboBox2 = QComboBox()
        self.widgetComboBox2.addItems(["Only KB", "Only LLM", "Both"])
        # Sends the current index (position) of the selected item.
        self.widgetComboBox2.currentIndexChanged.connect( self.index_changed )
        # There is an alternate signal to send the text.
        self.widgetComboBox2.currentTextChanged.connect( self.text_changed )

        # Qline
        self.widgetTextbox = QTextEdit()
        self.widgetTextbox.setFixedHeight(200)
        self.widgetTextbox.setPlaceholderText("Enter your text")

        #widget.setReadOnly(True) # uncomment this to make it read-only

        # self.widgetTextbox.returnPressed.connect(self.return_pressed)
        # self.widgetTextbox.selectionChanged.connect(self.selection_changed)
        self.widgetTextbox.textChanged.connect(self.text_changed)
        # self.widgetTextbox.textEdited.connect(self.text_edited)

        self.textboxLeft = QTextEdit()
        self.textboxLeft.setReadOnly(True)
        self.textboxLeft.setPlaceholderText("Top N diseases")

        self.textboxRight = QTextEdit()
        self.textboxRight.setReadOnly(True)
        self.textboxRight.setPlaceholderText("Explanation about disease")

        self.textboxMiddle = QTextEdit()
        self.textboxMiddle.setReadOnly(True)
        self.textboxMiddle.setFixedHeight(50)
        self.textboxMiddle.setPlaceholderText("Performance")

        text_layout = QHBoxLayout()
        text_layout.addWidget(self.textboxLeft)
        text_layout.addWidget(self.textboxRight)

        self.imageLabel = QLabel()
        pixmap = QPixmap(LOGO_PATH)

        self.imageLabel.setPixmap(pixmap)
        scaled = pixmap.scaled(
            150, 60,                                   # target size
            Qt.AspectRatioMode.KeepAspectRatio,         # no distortion
            Qt.TransformationMode.SmoothTransformation  # high quality
        )
        self.imageLabel.setPixmap(scaled)
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Container + layout
        central = QWidget()
        layout1 = QVBoxLayout(central)
        layout1.addWidget(self.label)
        layout1.addWidget(self.widgetTextbox)
        layout1.addWidget(self.widgetBox1)
        layout1.addWidget(self.widgetBox2)
        layout1.addWidget(self.widgetComboBox1)
        layout1.addWidget(self.widgetComboBox2)
        layout1.addLayout(text_layout) 
        layout1.addWidget(self.textboxMiddle)
        layout1.addStretch(1)
        layout1.addWidget(self.imageLabel)  

        self.setCentralWidget(central)

    def show_state(self, s):
        print(s == Qt.CheckState.Checked.value)
        print(s)
    
    def index_changed(self, i): # i is an int
        print(i)

    def text_changed(self, s): # s is a str
        print(s)

    def return_pressed(self):
        print("Return pressed!")
        self.centralWidget().setText("BOOM!")

    def selection_changed(self):
        print("Selection changed")
        print(self.centralWidget().selectedText())

    def text_changed(self, s):
        print("Text changed...")
        print(s)

    def text_edited(self, s):
        print("Text edited...")
        print(s)

app = QApplication(sys.argv)
window = MainWindow()
window.showMaximized()
app.exec()


