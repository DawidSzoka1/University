from io import BytesIO

import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QInputDialog, QMessageBox

from convertImage import qpixmap_to_numpy, numpy_to_qpixmap
import matplotlib.pyplot as plt


class HistogramTransformation:
    def __init__(self, histogram_label, transform_label, histogram_label_transform):
        self.histogram_label = histogram_label
        self.transform_label = transform_label
        self.histogram_label_transform = histogram_label_transform

    def show_histogram(self, pixmap, target):
        arr = qpixmap_to_numpy(pixmap)
        plt.figure(figsize=(4, 3), dpi=100)
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            plt.hist(arr[:, :, i].flatten(), bins=256, color=color, alpha=0.5, label=f'{color.upper()} kanał')
        plt.title("Histogram RGB")
        plt.xlabel("Wartość piksela")
        plt.ylabel("Liczba pikseli")
        plt.legend()
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        histogram_pixmap = QPixmap()
        histogram_pixmap.loadFromData(buf.read(), 'PNG')
        target.setPixmap(histogram_pixmap)

    def histogram_equalization(self, pixmap):
        img = qpixmap_to_numpy(pixmap)
        img_eq = np.zeros_like(img)

        for i in range(3):
            channel = img[..., i]
            hist, bins = np.histogram(channel.flatten(), bins=256, range=[0, 256])
            cdf = hist.cumsum()
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            cdf_normalized = cdf_normalized.astype(np.uint8)
            img_eq[..., i] = cdf_normalized[channel]
        result = numpy_to_qpixmap(img_eq)
        self.transform_label.setPixmap(result)
        self.show_histogram(result, self.histogram_label_transform)

    def histogram_scaling(self, pixmap, frame):
        L_min, ok1 = QInputDialog.getInt(frame, 'Podaj Lmin', 'Wartość Lmin:', 0, 0, 255, 1)
        L_max, ok2 = QInputDialog.getInt(frame, 'Podaj Lmax', 'Wartość Lmax:', 255, 0, 255, 1)

        if not ok1 or not ok2:
            QMessageBox.warning(frame, "Błąd", "Nie podano poprawnych wartości Lmin i Lmax!")
            return
        arr = qpixmap_to_numpy(pixmap)
        img_scaled = np.zeros_like(arr)

        for i in range(3):  # R, G, B
            channel = arr[..., i].astype(np.float32)
            min_val = channel.min()
            max_val = channel.max()
            if max_val - min_val == 0:
                img_scaled[..., i] = channel.astype(np.uint8)
            else:
                img_scaled[..., i] = (((channel - min_val) * (L_max - L_min)) / (max_val - min_val) + L_min).astype(
                    np.uint8)

        result = numpy_to_qpixmap(img_scaled)
        self.transform_label.setPixmap(result)
        self.show_histogram(result, self.histogram_label_transform)
