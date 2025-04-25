import math
from PIL import Image
from PyQt5.QtGui import QPixmap
from convertImage import qpixmap_to_numpy, numpy_to_qpixmap
import numpy as np


class MixingTransformation:
    def __init__(self, transformation_image):
        self.transformation_image = transformation_image

    def add_images(self, arrA, arrB):
        return np.clip(arrA + arrB, 0, 1)

    def subtractive(self, arrA, arrB):
        return np.clip(arrA + arrB - 1, 0, 1)

    def difference(self, arrA, arrB):
        return np.clip(np.abs(arrA - arrB), 0, 1)

    def multiply(self, arrA, arrB):
        return np.clip(arrA * arrB, 0, 1)

    def screen_mode(self, arrA, arrB):
        return np.clip(1 - (1 - arrA) * (1 - arrB), 0, 1)

    def negation(self, arrA, arrB):
        return np.clip(1 - np.abs(1 - arrA - arrB), 0, 1)

    def darken(self, arrA, arrB):
        return np.clip(np.where(arrA < arrB, arrA, arrB), 0, 1)

    def lighten(self, arrA, arrB):
        return np.clip(np.where(arrA > arrB, arrA, arrB), 0, 1)

    def exclusion(self, arrA, arrB):
        return np.clip(arrA + arrB - 2 * arrA * arrB, 0, 1)

    def overlay(self, arrA, arrB):
        return np.clip(np.where(arrA < 0.5, 2 * arrA * arrB, 1 - 2 * (1 - arrA) * (1 - arrB)), 0, 1)

    def hard_light(self, arrA, arrB):
        return np.clip(np.where(arrB < 0.5, 2 * arrA * arrB, 1 - 2 * (1 - arrA) * (1 - arrB)), 0, 1)

    def soft_light(self, arrA, arrB):
        return np.clip(np.where(arrB < 0.5, 2 * arrA * arrB + arrA ** 2 * (1 - 2 * arrB),
                                np.sqrt(arrA) * (2 * arrB - 1) + 2 * arrA * (1 - arrB)), 0, 1)

    def color_dodge(self, arrA, arrB):
        return np.clip(arrA / (1 - arrB + 1e-5), 0, 1)

    def color_burn(self, arrA, arrB):
        return np.clip(1 - (1 - arrA) / (arrB + 1e-5), 0, 1)

    def reflect(self, arrA, arrB):
        return np.clip(arrA ** 2 / (1 - arrB + 1e-5), 0, 1)

    def transparency(self, arrA, arrB, alpha=0.5):
        return np.clip((1 - alpha) * arrB + alpha * arrA, 0, 1)

    def transform(self, image_pathA, image_pathB, row=0, col=0):
        arr_A = qpixmap_to_numpy(QPixmap(image_pathA)).astype(np.float32) / 255.0
        arr_B = qpixmap_to_numpy(QPixmap(image_pathB)).astype(np.float32) / 255.0

        common_height = min(arr_A.shape[1], arr_B.shape[1])
        common_width = min(arr_A.shape[0], arr_B.shape[0])
        func = [
            [self.add_images, self.subtractive, self.difference, self.multiply],
            [self.screen_mode, self.negation, self.darken, self.lighten],
            [self.exclusion, self.overlay, self.hard_light, self.soft_light],
            [self.color_dodge, self.color_burn, self.reflect, self.transparency]
        ]
        func_use = func[row][col]

        arr_A = arr_A[:common_width, :common_height]
        arr_B = arr_B[:common_width, :common_height]

        result = np.clip(func_use(arr_A, arr_B) * 255, 0, 255).astype(np.uint8)
        self.transformation_image.setPixmap(numpy_to_qpixmap(result))
