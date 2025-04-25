import math
from PIL import Image
from PyQt5.QtGui import QPixmap
from convertImage import qpixmap_to_numpy, numpy_to_qpixmap
import numpy as np


class MixingTransformation:
    def __init__(self, transformation_image):
        self.transformation_image = transformation_image

    def add_images(self, arrA, arrB):
        return arrA + arrB

    def subtractive(self, arrA, arrB):
        return arrA + arrB - 1

    def difference(self, arrA, arrB):
        return abs(arrA - arrB)

    def multiply(self, arrA, arrB):
        return arrA * arrB

    def screen_mode(self, arrA, arrB):
        return 1 - (1 - arrA) * (1 - arrB)

    def negation(self, arrA, arrB):
        return 1 - abs(1 - arrA - arrB)

    def darken(self, arrA, arrB):
        return arrA if arrA < arrB else arrB

    def lighten(self, arrA, arrB):
        return arrA if arrA > arrB else arrB

    def exclusion(self, arrA, arrB):
        return arrA + arrB - 2 * arrA * arrB

    def overlay(self, arrA, arrB):
        return 2 * arrA * arrB if arrA < 0.5 else 1 - 2 * (1 - arrA) * (1 - arrB)

    def hard_light(self, arrA, arrB):
        return 2 * arrA * arrB if arrB < 0.5 else 1 - 2 * (1 - arrA) * (1 - arrB)

    def soft_light(self, arrA, arrB):
        return 2 * arrA * arrB + arrA ** 2 * (1 - 2 * arrB) if arrB < 0.5 \
            else math.sqrt(arrA) * (2 * arrB - 1) + 2 * arrA * (1 - arrB)

    def color_dodge(self, arrA, arrB):
        return arrA / (1 - arrB)

    def color_burn(self, arrA, arrB):
        return 1 - (1 - arrA) / arrB

    def reflect(self, arrA, arrB):
        return arrA ** 2 / (1 - arrB)

    def transparency(self, arrA, arrB, alpha = 0.5):
        return (1 - alpha) * arrB + alpha * arrA

    def transform(self, image_pathA, image_pathB, row=0, col=0):
        arr_A = qpixmap_to_numpy(QPixmap(image_pathA)).astype(np.float32) / 255.0
        arr_B = qpixmap_to_numpy(QPixmap(image_pathB)).astype(np.float32) / 255.0
        print(f"ROw : {row}, Col : {col}")
        common_height = min(arr_A.shape[1], arr_B.shape[1])
        common_width = min(arr_A.shape[0], arr_B.shape[0])
        func = [
            [self.add_images, self.subtractive, self.difference, self.multiply],
            [self.screen_mode, self.negation, self.darken, self.lighten],
            [self.exclusion, self.overlay, self.hard_light, self.soft_light],
            [self.color_dodge, self.color_burn, self.reflect, self.transparency]
        ]
        func_use = func[row][col]
        print(func_use)
        arr_A = arr_A[:common_width, :common_height]
        arr_B = arr_B[:common_width, :common_height]

        result = np.clip(func_use(arr_A, arr_B) * 255, 0, 255).astype(np.uint8)
        self.transformation_image.setPixmap(numpy_to_qpixmap(result))
