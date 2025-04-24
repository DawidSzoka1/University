import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from convertImage import qpixmap_to_numpy, numpy_to_qpixmap

class PowerTransformation:
    def __init__(self, image_transformation):
        self.image_transformation = image_transformation
    def power_transformation(self, pixmap, gamma):
        np_img = qpixmap_to_numpy(pixmap)

        for i in range(3):  # R, G, B
            np_img[..., i] = np.clip((np_img[..., i] / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)

        transformed = numpy_to_qpixmap(np_img)
        self.image_transformation.setPixmap(transformed)

    def brightness(self, pixmap, gamma=0.5):
        self.power_transformation(pixmap, gamma)

    def darken(self, pixmap, gamma=2):
        self.power_transformation(pixmap, gamma)
