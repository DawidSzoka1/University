from scipy.ndimage import minimum_filter, maximum_filter, median_filter
from convertImage import qpixmap_to_numpy, numpy_to_qpixmap
import numpy as np


class StatisticsTransformation:
    def __init__(self, image_transformation):
        self.image_transformation = image_transformation

    def statistics_transformation(self, pixmap, filter_type='median', kernel_size=3):
        image = qpixmap_to_numpy(pixmap)
        output = np.zeros_like(image)
        if filter_type == 'median':
            filter_func = median_filter
        elif filter_type == 'min':
            filter_func = minimum_filter
        else:
            filter_func = maximum_filter
        for i in range(3):
            output[..., i] = filter_func(image[..., i], size=kernel_size)

        self.image_transformation.setPixmap(numpy_to_qpixmap(output))
