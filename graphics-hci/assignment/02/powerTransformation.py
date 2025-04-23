import numpy as np
from PyQt5.QtGui import QPixmap, QImage

class PowerTransformation:
    def __init__(self, image_transformation):
        self.image_transformation = image_transformation

    def qpixmap_to_numpy(self, pixmap):
        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
        return arr.copy()

    def numpy_to_qpixmap(self, arr):
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        image = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        return QPixmap.fromImage(image.copy())

    def power_transformation(self, pixmap, gamma):
        np_img = self.qpixmap_to_numpy(pixmap)

        for i in range(3):  # R, G, B
            np_img[..., i] = np.clip((np_img[..., i] / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)

        transformed = self.numpy_to_qpixmap(np_img)
        self.image_transformation.setPixmap(transformed)

    def brightness(self, pixmap, gamma=0.5):
        self.power_transformation(pixmap, gamma)

    def darken(self, pixmap, gamma=2):
        self.power_transformation(pixmap, gamma)
