import numpy as np
from PyQt5.QtGui import QImage, QPixmap

def qpixmap_to_numpy(pixmap):
    image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
    return arr.copy()

def numpy_to_qpixmap(arr):
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    image = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())