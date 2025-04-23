from PyQt5.QtGui import QImage, QPixmap, QColor


class LineTransformation:
    def __init__(self, image_transformation):
        self.image_transformation = image_transformation

    def brightness(self, pixmap, amount=30):
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format_RGB32)
        for y in range(image.height()):
            for x in range(image.width()):
                color = QColor(image.pixel(x, y))
                r = min(color.red() + amount, 255)
                g = min(color.green() + amount, 255)
                b = min(color.blue() + amount, 255)
                image.setPixel(x, y, QColor(r, g, b).rgb())

        self.image_transformation.setPixmap(QPixmap.fromImage(image))


    def negative(self, pixmap):
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format_RGB32)
        for y in range(image.height()):
            for x in range(image.width()):
                color = QColor(image.pixel(x, y))
                r = 255 - color.red()
                g = 255 - color.green()
                b = 255 - color.blue()
                image.setPixel(x, y, QColor(r, g, b).rgb())

        self.image_transformation.setPixmap(QPixmap.fromImage(image))

    def darken(self, pixmap, amount=30):
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format_RGB32)
        for y in range(image.height()):
            for x in range(image.width()):
                color = QColor(image.pixel(x, y))
                r = max(color.red() - amount, 0)
                g = max(color.red() - amount, 0)
                b = max(color.red() - amount, 0)
                image.setPixel(x, y, QColor(r, g, b).rgb())

        self.image_transformation.setPixmap(QPixmap.fromImage(image))
