from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtGui import QPixmap

def qpixmap_to_pil(pixmap):
    buffer = pixmap.toImage().bits().asstring(pixmap.width() * pixmap.height() * 4)
    image = Image.frombytes("RGBA", (pixmap.width(), pixmap.height()), buffer)
    return image

def pil_to_qpixmap(pil_image):
    return QPixmap.fromImage(ImageQt(pil_image))