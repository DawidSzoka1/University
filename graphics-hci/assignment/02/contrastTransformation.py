from PyQt5.QtWidgets import QInputDialog, QMessageBox

from convertImage import qpixmap_to_numpy, numpy_to_qpixmap
import numpy as np


class ContrasTransformation:
    def __init__(self,  transformation_image):
        self.transformation_image = transformation_image


    def transform(self, pixmap, frame):
        contrast, ok1 = QInputDialog.getInt(frame,
                                            "Kontrast",
                                            "Wprowadz wartosc kontrastu od 0 do 126",
                                            50, 0, 126, 1)
        if not ok1:
            QMessageBox.warning(frame, "Podana zla wartosc kontrastu")
            return
        arr = qpixmap_to_numpy(pixmap)
        for i in range(3):
            arr[..., i] = np.clip(127/(127-contrast) * (arr[..., i].astype(np.int32) - contrast), 0, 255).astype(np.uint8)

        self.transformation_image.setPixmap(numpy_to_qpixmap(arr))