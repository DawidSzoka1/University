import numpy as np
from PyQt5.QtGui import QImage, QPixmap

class LineTransformation:
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

    def brightness(self, pixmap, amount=50):
        arr = self.qpixmap_to_numpy(pixmap)
        for i in range(3):  # R, G, B
            arr[..., i] = np.clip(arr[..., i] + amount, 0, 255)
        result = self.numpy_to_qpixmap(arr)
        self.image_transformation.setPixmap(result)

    def darken(self, pixmap, amount=50):
        arr = self.qpixmap_to_numpy(pixmap)
        for i in range(3):  # R, G, B
            arr[..., i] = np.clip(arr[..., i] - amount, 0, 255)
        result = self.numpy_to_qpixmap(arr)
        self.image_transformation.setPixmap(result)

    def negative(self, pixmap):
        arr = self.qpixmap_to_numpy(pixmap)
        arr[..., :3] = 255 - arr[..., :3]  # tylko R, G, B
        result = self.numpy_to_qpixmap(arr)
        self.image_transformation.setPixmap(result)
