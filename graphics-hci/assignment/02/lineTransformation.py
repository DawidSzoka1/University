import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QInputDialog, QMessageBox

from convertImage import qpixmap_to_numpy, numpy_to_qpixmap


class LineTransformation:
    def __init__(self, image_transformation):
        self.image_transformation = image_transformation

    def brightness(self, pixmap, frame):
        amount, ok1 = QInputDialog.getInt(frame, "Rozjaśnienie", "moc", 20, 0, 200, 1)
        if not ok1:
            QMessageBox.warning(frame, "Błąd", "Nie podano poprawnych wartości rozjaśnienia")
            return
        arr = qpixmap_to_numpy(pixmap)
        for i in range(3):  # R, G, B
            arr[..., i] = np.clip(arr[..., i].astype(np.int16) + amount, 0, 255).astype(np.uint8)

        result = numpy_to_qpixmap(arr)
        self.image_transformation.setPixmap(result)

    def darken(self, pixmap, frame):
        amount, ok1 = QInputDialog.getInt(frame, "Przyciemnienie", "moc", 20, 0, 200, 1)
        if not ok1:
            QMessageBox.warning(frame, "Błąd", "Nie podano poprawnych wartości przyciemnienia")
            return
        arr = qpixmap_to_numpy(pixmap)
        for i in range(3):  # R, G, B
            arr[..., i] = np.clip(arr[..., i].astype(np.int16) - amount, 0, 255).astype(np.uint8)

        result = numpy_to_qpixmap(arr)
        self.image_transformation.setPixmap(result)

    def negative(self, pixmap):
        arr = qpixmap_to_numpy(pixmap)
        for i in range(3):  # R, G, B
            arr[..., i] = np.clip(255 - arr[..., i].astype(np.int16), 0, 255).astype(np.uint8)
        result = numpy_to_qpixmap(arr)
        self.image_transformation.setPixmap(result)
