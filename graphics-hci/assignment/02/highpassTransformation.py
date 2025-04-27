import numpy as np
from PyQt5.QtGui import QPixmap
from PIL import Image
from PyQt5.QtWidgets import QInputDialog, QMessageBox


class HighPassTransformation:
    def __init__(self, image_transformation):
        self.image_transformation = image_transformation
        roberts_h = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
        roberts_v = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])

        prewitt_h = np.array([[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]])

        prewitt_v = np.array([[1, 1, 1],
                              [0, 0, 0],
                              [-1, -1, -1]])

        sobel_h = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

        sobel_v = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

        laplace_1 = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]])
        laplace_2 = np.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
        laplace_3 = np.array([[-2, 1, -2],
                              [1, 4, 1],
                              [-2, 1, -2]])

        self.kernel = {
            "sobel_h": sobel_h,
            "sobel_v": sobel_v,
            "prewitt_h": prewitt_h,
            "prewitt_v": prewitt_v,
            "roberts_h": roberts_h,
            "roberts_v": roberts_v,
            "laplace_1": laplace_1,
            "laplace_2": laplace_2,
            "laplace_3": laplace_3,
        }

    def choose_laplace(self, frame, image_path):
        laplace, ok1 = QInputDialog.getInt(frame, "Laplace", "wybierz maske laplace(1-3)", 1, 1, 3, 1)
        if not ok1:
            QMessageBox.warning(frame, "Błąd", "Nie wybrano poprawnie maski")
            return
        self.transform(image_path, "laplace_" + str(laplace))

    def transform(self, image_path, kernel_type="sobel_h"):
        image = Image.open(image_path).convert("RGB")
        pixels = image.load()
        h, w = image.height, image.width
        output = Image.new("RGB", (w, h))
        output_pixels = output.load()
        kerner = self.kernel.get(kernel_type)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                acc_r = 0
                acc_g = 0
                acc_b = 0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        r, g, b = pixels[x + kx, y + ky]
                        kernel_value = kerner[ky + 1][kx + 1]
                        acc_r += r * kernel_value
                        acc_g += g * kernel_value
                        acc_b += b * kernel_value
                acc_r = max(0, min(255, abs(acc_r)))
                acc_g = max(0, min(255, abs(acc_g)))
                acc_b = max(0, min(255, abs(acc_b)))
                output_pixels[x, y] = (acc_r, acc_g, acc_b)
        output.save("transform.png")

        self.image_transformation.setPixmap(QPixmap("transform.png"))
