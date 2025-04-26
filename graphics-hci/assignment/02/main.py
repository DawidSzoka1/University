import sys
from lowpassTransformation import LowPassTransformation
from contrastTransformation import ContrasTransformation
from histogramTransformation import HistogramTransformation
from lineTransformation import LineTransformation
from statisticsTransformation import StatisticsTransformation
from highpassTransformation import HighPassTransformation
from mixingTransformation import MixingTransformation
from powerTransformation import PowerTransformation
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox, QGridLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from functools import partial


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.mixing_add = QPushButton("Suma", self)
        self.mixing_sub = QPushButton("Odejmowanie", self)
        self.mixing_diff = QPushButton("Różnica", self)
        self.mixing_multi = QPushButton("Mnozenie", self)
        self.mixing_screen = QPushButton("Mnożenie odwrotności", self)
        self.mixing_negation = QPushButton("Negacja", self)
        self.mixing_darken = QPushButton("Ciemniejsze", self)
        self.mixing_lighten = QPushButton("Jaśniejsze", self)
        self.mixing_exclusion = QPushButton("Wyłaczenie", self)
        self.mixing_overlay = QPushButton("Nakładka", self)
        self.mixing_hardLight = QPushButton("Ostre światło", self)
        self.mixing_softLight = QPushButton("Łagodne światło", self)
        self.mixing_colorDodge = QPushButton("Rozcieńczenie", self)
        self.mixing_colorBurn = QPushButton("Wypalanie", self)
        self.mixing_reflect = QPushButton("Reflect mode", self)
        self.mixing_transparency = QPushButton("Przezroczystość", self)

        self.mixing_images = [
            [self.mixing_add, self.mixing_sub, self.mixing_diff, self.mixing_multi],
            [self.mixing_screen, self.mixing_negation, self.mixing_darken, self.mixing_lighten],
            [self.mixing_exclusion, self.mixing_overlay, self.mixing_hardLight, self.mixing_softLight],
            [self.mixing_colorDodge, self.mixing_colorBurn, self.mixing_reflect, self.mixing_transparency]
        ]

        self.image_path = None
        self.setWindowTitle("Transformacje obrazu")
        self.setGeometry(100, 100, 1300, 800)
        self.is_image_loaded = False
        self.image_label = QLabel("Obraz 1")
        self.image_label.setFixedSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.setScaledContents(True)

        self.second_image_path = None
        self.second_image_label = QLabel("Obraz 2")
        self.second_image_label.setFixedSize(400, 300)
        self.second_image_label.setStyleSheet("border: 1px solid gray;")
        self.second_image_label.setScaledContents(True)

        self.image_transformation = QLabel("Transformowany obraz")
        self.image_transformation.setFixedSize(400, 300)
        self.image_transformation.setStyleSheet("border: 1px solid gray;")
        self.image_transformation.setScaledContents(True)

        self.histogram_label = QLabel("Histogram")
        self.histogram_label.setFixedSize(400, 300)
        self.histogram_label.setStyleSheet("border: 1px solid gray;")
        self.histogram_label.setScaledContents(True)

        # Przycisk ładowania
        self.button_load = QPushButton("Wybierz zdjęcie")
        self.button_load.clicked.connect(self.choose_photo)

        self.button_load_2 = QPushButton("Wybierz drugie zdjęcie")
        self.button_load_2.clicked.connect(self.choose_second_photo)

        self.button_save_1 = QPushButton("Zapisz obraz Transformowany")

        image_layout = QGridLayout()
        image_layout.addWidget(self.image_label, 0, 0)
        image_layout.addWidget(self.second_image_label, 0, 1)
        image_layout.addWidget(self.image_transformation, 1, 0)
        image_layout.addWidget(self.histogram_label, 1, 1)
        image_layout.addWidget(self.button_load, 2, 0)
        image_layout.addWidget(self.button_load_2, 2, 1)
        image_layout.addWidget(self.button_save_1, 3, 0)

        self.lineTransformation = LineTransformation(self.image_transformation)
        self.powerTransformation = PowerTransformation(self.image_transformation)
        self.histogramTransformation = HistogramTransformation(self.second_image_label, self.image_transformation,
                                                               self.histogram_label)
        self.statisticsTransformation = StatisticsTransformation(self.image_transformation)
        self.hightpassTransformation = HighPassTransformation(self.image_transformation)
        self.mixingTransformation = MixingTransformation(self.image_transformation)
        self.contrastTransformation = ContrasTransformation(self.image_transformation)
        self.lowPassTransformation = LowPassTransformation(self.image_transformation)
        # Grupy przycisków
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.group_linear_transformation())
        control_layout.addWidget(self.group_power_transformation())
        control_layout.addWidget(self.group_mixing())
        control_layout.addWidget(self.group_contrast())
        control_layout.addWidget(self.group_histogram())
        control_layout.addWidget(self.group_low_pass())
        control_layout.addWidget(self.group_high_pass())
        control_layout.addWidget(self.group_statistics())
        control_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)

    def choose_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik", "", "Obrazy (*.png *.jpg *.jpeg *.bmp)")
        if path:
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap)
            self.image_path = path
            self.add_clicked()
        if self.image_path and self.second_image_path:
            self.add_mixing_clicked()

    def choose_second_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz drugie zdjęcie", "", "Obrazy (*.png *.jpg *.jpeg *.bmp)")
        if path:
            pixmap = QPixmap(path)
            self.second_image_path = path
            self.second_image_label.setPixmap(pixmap)
        if self.image_path and self.second_image_path:
            self.add_mixing_clicked()

    def save_image(self, image):
        if image.pixmap() is None or image.pixmap().isNull():
            QMessageBox.warning(self, "Błąd", "Nie zostalo wykonane jeszcze przeksztalcenie")
            return

        image = image.pixmap().toImage()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg)")
        if filePath == "":
            return

        image.save(filePath)

    def add_mixing_clicked(self):

        for i in range(4):
            for j in range(4):
                self.mixing_images[i][j].clicked.connect(
                    partial(self.mixingTransformation.transform, self.image_path, self.second_image_path, i, j)
                )

    def add_clicked(self):
        self.button_save_1.clicked.connect(lambda: self.save_image(self.image_transformation))

        self.brightnes.clicked.connect(lambda: self.lineTransformation.brightness(self.image_label.pixmap(), self))
        self.darkening.clicked.connect(lambda: self.lineTransformation.darken(self.image_label.pixmap(), self))
        self.negative.clicked.connect(lambda: self.lineTransformation.negative(self.image_label.pixmap()))

        self.power_brightnes.clicked.connect(lambda: self.powerTransformation.brightness(self.image_label.pixmap()))
        self.power_darkening.clicked.connect(lambda: self.powerTransformation.darken(self.image_label.pixmap()))

        self.histogram.clicked.connect(lambda: self.histogramTransformation.show_histogram(self.image_label.pixmap(),
                                                                                           self.second_image_label))

        self.histogram_equalization.clicked.connect(
            lambda: self.histogramTransformation.histogram_equalization(self.image_label.pixmap()))
        self.histogram_scale.clicked.connect(
            lambda: self.histogramTransformation.histogram_scaling(self.image_label.pixmap(), self))

        self.contrast_button.clicked.connect(
            lambda: self.contrastTransformation.transform(self.image_label.pixmap(), self)
        )

        self.low_pass.clicked.connect(
            lambda: self.lowPassTransformation.transform(self.image_label.pixmap())
        )

        self.sobel_horizontal.clicked.connect(
            lambda: self.hightpassTransformation.transform(self.image_path)
        )
        self.sobel_vertical.clicked.connect(
            lambda: self.hightpassTransformation.transform(self.image_path, "sobel_v")
        )
        self.prewitt_horizontal.clicked.connect(
            lambda: self.hightpassTransformation.transform(self.image_path, "prewitt_h")
        )
        self.prewitt_vertical.clicked.connect(
            lambda: self.hightpassTransformation.transform(self.image_path, "prewitt_v")
        )
        self.roberts_horizontal.clicked.connect(
            lambda: self.hightpassTransformation.transform(self.image_path, "roberts_h")
        )
        self.roberts_vertical.clicked.connect(
            lambda: self.hightpassTransformation.transform(self.image_path, "roberts_v")
        )
        self.laplace.clicked.connect(
            lambda: self.hightpassTransformation.choose_laplace(self, self.image_path)
        )

        self.filtr_min.clicked.connect(lambda:
                                       self.statisticsTransformation.statistics_transformation(
                                           self.image_label.pixmap(), "min"))
        self.filtr_max.clicked.connect(lambda:
                                       self.statisticsTransformation.statistics_transformation(
                                           self.image_label.pixmap(), "max"))
        self.filtr_median.clicked.connect(lambda:
                                          self.statisticsTransformation.statistics_transformation(
                                              self.image_label.pixmap(), "median"))

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
    def group_mixing(self):
        box = QGroupBox("Mieszanie obrazów")
        grid = QGridLayout()

        for i in range(16):
            row = i // 4
            col = i % 4
            grid.addWidget(self.mixing_images[row][col], row, col)

        box.setLayout(grid)
        return box

    # Grupa: Modyfikacji kontrastu
    def group_contrast(self):
        box = QGroupBox("Modyfikacja kontrastu")
        layout = QVBoxLayout()
        self.contrast_button = QPushButton("Obrazu barwnego wariant 1 a)", self)
        layout.addWidget(self.contrast_button)
        box.setLayout(layout)
        return box

    # Grupa: Histogramy
    def group_histogram(self):
        box = QGroupBox("Histogramy")
        layout = QVBoxLayout()
        self.histogram = QPushButton("Generuj histogram RGB", self)
        layout.addWidget(self.histogram)

        self.histogram_equalization = QPushButton("Wyrównanie histogramu", self)
        layout.addWidget(self.histogram_equalization)

        self.histogram_scale = QPushButton("Skalowanie histogramu", self)
        layout.addWidget(self.histogram_scale)
        box.setLayout(layout)
        return box

    # Grupa: Filtry dolnoprzepustowe
    def group_low_pass(self):
        box = QGroupBox("Filtry dolnoprzepustowe")
        layout = QVBoxLayout()
        self.low_pass = QPushButton("Filtr dolnoprzepustowy", self)
        layout.addWidget(self.low_pass)
        box.setLayout(layout)
        return box

    # Grupa: Filtry górnoprzepustowe
    def group_high_pass(self):
        box = QGroupBox("Filtry górnoprzepustowe")
        grid = QGridLayout()
        self.roberts_horizontal = QPushButton("Roberts (poziomy)", self)
        grid.addWidget(self.roberts_horizontal, 0, 0)
        self.roberts_vertical = QPushButton("Roberts (pionowy)", self)
        grid.addWidget(self.roberts_vertical, 0, 1)
        self.prewitt_horizontal = QPushButton("Prewitt (poziomy)", self)
        grid.addWidget(self.prewitt_horizontal, 1, 0)
        self.prewitt_vertical = QPushButton("Prewitt (pionowy)", self)
        grid.addWidget(self.prewitt_vertical, 1, 1)
        self.sobel_horizontal = QPushButton("Sobel (poziomy)", self)
        grid.addWidget(self.sobel_horizontal, 2, 0)
        self.sobel_vertical = QPushButton("Sobel (pionowy)", self)
        grid.addWidget(self.sobel_vertical, 2, 1)
        self.laplace = QPushButton("Laplace (maski 1-3)", self)
        grid.addWidget(self.laplace, 3, 0)
        box.setLayout(grid)
        return box

    # Grupa: Filtry statystyczne
    def group_statistics(self):
        box = QGroupBox("Filtry statystyczne")
        layout = QVBoxLayout()
        self.filtr_min = QPushButton("Filtr min", self)
        layout.addWidget(self.filtr_min)
        self.filtr_max = QPushButton("Filtr max", self)
        layout.addWidget(self.filtr_max)
        self.filtr_median = QPushButton("Filtr medianowy", self)
        layout.addWidget(self.filtr_median)
        box.setLayout(layout)
        return box


# Start
if __name__ == "__main__":
    app = QApplication(sys.argv)
    okno = MainWindow()
    okno.show()
    sys.exit(app.exec_())
