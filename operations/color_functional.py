import cv2
import numpy as np
from . import utils

device = utils.device


class color_transform(object):
    def __init__(self, image):
        self.image = image


class normalization_cpu(color_transform):
    def __init___(self, image):
        color_transform.__init__(self, image)

    def __call__(self, *args, **kwargs):
        norm_img = np.zeros((self.image.shape[0], self.image.shape[1]))
        return cv2.normalize(self.image, norm_img, 0, 255, cv2.NORM_MINMAX)

