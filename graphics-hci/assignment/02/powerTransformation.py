from PIL import Image
from PyQt5.QtGui import QImage, QPixmap, QColor
from util import qpixmap_to_pil, pil_to_qpixmap


class PowerTransformation:
    def __init__(self, image_transformation):
        self.image_transformation = image_transformation

    def power_transformation(self, pixmap, gamma):
        image = qpixmap_to_pil(pixmap).convert("RGB")
        result_image = Image.new('RGBA', image.size)
        w,h = image.size
        for y in range(w):
            for x in range(h):
                r,g,b = image.getpixel((x, y))
                r = int(((r / 255) ** gamma) * 255)
                g = int(((g / 255) ** gamma) * 255)
                b = int(((b / 255) ** gamma) * 255)
                result_image.putpixel((x, y), (r, g, b))

        self.image_transformation.setPixmap(pil_to_qpixmap(result_image))

    def brightness(self, pixmap, gamma=0.5):
        self.power_transformation(pixmap, gamma)

    def darken(self, pixmap, gamma=2):
        self.brightness(pixmap, gamma)
