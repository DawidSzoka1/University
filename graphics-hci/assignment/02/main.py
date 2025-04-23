import sys
from lineTransformation import LineTransformation
from powerTransformation import PowerTransformation
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox
)
from PyQt5.QtGui import QPixmap

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Transformacje obrazu")
        self.setGeometry(100, 100, 1000, 600)
        self.is_image_loaded = False
        # Obrazy
        self.image_label = QLabel("Obraz")
        self.image_label.setFixedSize(500, 400)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.setScaledContents(True)

        self.image_transformation = QLabel("Transformowany obraz")
        self.image_transformation.setFixedSize(500, 400)
        self.image_transformation.setStyleSheet("border: 1px solid gray;")
        self.image_transformation.setScaledContents(True)
        self.lineTransformation = LineTransformation(self.image_transformation)
        self.powerTransformation = PowerTransformation(self.image_transformation)

        # Przycisk ładowania
        self.button_load = QPushButton("Wybierz zdjęcie")
        self.button_load.clicked.connect(self.choose_photo)

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.image_transformation)
        image_layout.addWidget(self.button_load)

        # Grupy przycisków
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.group_linear_transformation())
        control_layout.addWidget(self.group_power_transformation())
        control_layout.addWidget(self.grupa_mieszanie())
        control_layout.addWidget(self.grupa_histogramy())
        control_layout.addWidget(self.grupa_filtry_dolno())
        control_layout.addWidget(self.grupa_filtry_gorno())
        control_layout.addWidget(self.grupa_filtry_statystyczne())
        control_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)

    def choose_photo(self):
        sciezka, _ = QFileDialog.getOpenFileName(self, "Wybierz plik", "", "Obrazy (*.png *.jpg *.jpeg *.bmp)")
        if sciezka:
            pixmap = QPixmap(sciezka)
            self.image_label.setPixmap(pixmap)
            self.add_clicked()


    def add_clicked(self):
        self.brightnes.clicked.connect(lambda: self.lineTransformation.brightness(self.image_label.pixmap()))
        self.darkening.clicked.connect(lambda: self.lineTransformation.darken(self.image_label.pixmap()))
        self.negative.clicked.connect(lambda: self.lineTransformation.negative(self.image_label.pixmap()))

        self.power_brightnes.clicked.connect(lambda: self.powerTransformation.brightness(self.image_label.pixmap()))
        self.power_darkening.clicked.connect(lambda: self.powerTransformation.darken(self.image_label.pixmap()))

    # Grupa: Transformacje liniowe
    def group_linear_transformation(self):
        box = QGroupBox("Transformacje liniowe")
        layout = QVBoxLayout()
        self.brightnes = QPushButton("Rozjaśnienie", self)
        layout.addWidget(self.brightnes)
        self.darkening = QPushButton("Przyciemnienie", self)
        layout.addWidget(self.darkening)
        self.negative = QPushButton("Negatyw", self)
        layout.addWidget(self.negative)
        box.setLayout(layout)
        return box

    # Grupa: Transformacje potęgowe
    def group_power_transformation(self):
        box = QGroupBox("Transformacje potęgowe")
        layout = QVBoxLayout()
        self.power_brightnes = QPushButton("Rozjaśnienie(potęgowe)", self)
        layout.addWidget(self.power_brightnes)
        self.power_darkening = QPushButton("Przyciemnienie(potęgowe)", self)
        layout.addWidget(self.power_darkening)
        box.setLayout(layout)
        return box

    # Grupa: Mieszanie obrazów
    def grupa_mieszanie(self):
        box = QGroupBox("Mieszanie obrazów")
        layout = QVBoxLayout()
        for i in range(1, 5):  # Na razie 4, można dodać więcej
            layout.addWidget(QPushButton(f"Mieszaj {i}"))
        box.setLayout(layout)
        return box

    # Grupa: Histogramy
    def grupa_histogramy(self):
        box = QGroupBox("Histogramy")
        layout = QVBoxLayout()
        self.histogram = QPushButton("Generuj histogram RGB")
        layout.addWidget(self.histogram)
        layout.addWidget(QPushButton("Wyrównanie histogramu"))
        layout.addWidget(QPushButton("Skalowanie histogramu"))
        box.setLayout(layout)
        return box

    # Grupa: Filtry dolnoprzepustowe
    def grupa_filtry_dolno(self):
        box = QGroupBox("Filtry dolnoprzepustowe")
        layout = QVBoxLayout()
        layout.addWidget(QPushButton("Filtr dolnoprzepustowy"))
        box.setLayout(layout)
        return box

    # Grupa: Filtry górnoprzepustowe
    def grupa_filtry_gorno(self):
        box = QGroupBox("Filtry górnoprzepustowe")
        layout = QVBoxLayout()
        layout.addWidget(QPushButton("Roberts (poziomy)"))
        layout.addWidget(QPushButton("Roberts (pionowy)"))
        layout.addWidget(QPushButton("Prewitt (poziomy)"))
        layout.addWidget(QPushButton("Prewitt (pionowy)"))
        layout.addWidget(QPushButton("Sobel (poziomy)"))
        layout.addWidget(QPushButton("Sobel (pionowy)"))
        layout.addWidget(QPushButton("Laplace (maski 1–3)"))
        box.setLayout(layout)
        return box

    # Grupa: Filtry statystyczne
    def grupa_filtry_statystyczne(self):
        box = QGroupBox("Filtry statystyczne")
        layout = QVBoxLayout()
        layout.addWidget(QPushButton("Filtr min"))
        layout.addWidget(QPushButton("Filtr max"))
        layout.addWidget(QPushButton("Filtr medianowy"))
        box.setLayout(layout)
        return box

# Start
if __name__ == "__main__":
    app = QApplication(sys.argv)
    okno = MainWindow()
    okno.show()
    sys.exit(app.exec_())
