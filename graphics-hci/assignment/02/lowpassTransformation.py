from convertImage import qpixmap_to_numpy, numpy_to_qpixmap
import numpy as np


class LowPassTransformation:
    def __init__(self, transformation_image):
        self.transformation_image = transformation_image

    def transform(self, pixmap):
        arr = qpixmap_to_numpy(pixmap).astype(np.float32)
        h, w, ch = arr.shape
        result = np.zeros_like(arr)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                result[1:-1, 1:-1] += arr[1+dy:h-1+dy, 1+dx:w-1+dx]

        result[1:-1, 1:-1] /= 9.0
        result = np.clip(result, 0, 255).astype(np.uint8)
        self.transformation_image.setPixmap(numpy_to_qpixmap(result))
