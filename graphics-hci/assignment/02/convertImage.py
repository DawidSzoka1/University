import numpy as np
from PyQt5.QtGui import QImage, QPixmap

def qpixmap_to_numpy(pixmap):
    image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)  # 4 kanały: R, G, B, A
    arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
    return arr[..., :3].copy()  # zwróć tylko R, G, B (ignoruj Alfa)

def numpy_to_qpixmap(arr):
    h, w, ch = arr.shape
    assert ch == 3, "Array must have 3 channels (RGB)"
    arr_rgb = np.concatenate([arr, 255 * np.ones((h, w, 1), dtype=np.uint8)], axis=2)
    bytes_per_line = 4 * w
    image = QImage(arr_rgb.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())
