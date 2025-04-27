from convertImage import qpixmap_to_numpy, numpy_to_qpixmap
import numpy as np


class StatisticsTransformation:
    def __init__(self, image_transformation):
        self.image_transformation = image_transformation

    def statistics_transformation(self, pixmap, filter_type='median', kernel_size=3):
        image = qpixmap_to_numpy(pixmap)
        if filter_type == 'median':
            filter_func = np.median
        elif filter_type == 'min':
            filter_func = np.min
        else:
            filter_func = np.max
        pad = kernel_size // 2
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        output = np.zeros_like(image)

        height, width, channels = image.shape

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    region = padded_image[y:y+kernel_size, x:x+kernel_size, c]
                    output[y, x, c] = filter_func(region)
        self.image_transformation.setPixmap(numpy_to_qpixmap(output))
