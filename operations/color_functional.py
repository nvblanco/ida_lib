import cv2
import numpy as np
from . import utils
import torch
import kornia


device = utils.device


class color_transform(object):
    def __init__(self, image):
        if torch.is_tensor(image):
            image = kornia.tensor_to_image(image.byte())
        self.image = image


class normalization(color_transform):
    def __init___(self, image):
        color_transform.__init__(self, image)

    def __call__(self, *args, **kwargs):
        norm_img = np.zeros((self.image.shape[0], self.image.shape[1]))
        self.image = cv2.normalize(self.image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_img


class brightness_and_contrast(color_transform):
    def __init__(self, image, brightness,  normalized = True):
        color_transform.__init__(self, image)
        if brightness < 0 or brightness > 2:
            raise Exception("Brightness factor value must be between 0 and 2 (Received {}".format(brightness))
        if brightness < 0 or brightness > 2:
            raise Exception("Contrast factor value must be between 0 and 2 (Received {}".format(constrast))
        self.brigthness = brightness
        #self.contrast = constrast
        self.normalized = normalized

    def __call__(self, *args, **kwargs):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - self.brigthness
        v[v > lim] = 255
        v[v <= lim] += self.brigthness

        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)



