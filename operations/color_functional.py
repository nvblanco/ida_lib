import cv2
import numpy as np
from . import utils
import torch
import kornia
from image_augmentation import visualization


device = utils.device


class color_transform(object):
    def __init__(self, image, visualize = False):
        if isinstance(image, dict):
            self.data_type = 'dict'
            self.data = image
            image = image['image']
        else:
            self.data_type = 'image'
        if torch.is_tensor(image):
            self.image_type = 'tensor'
            image = kornia.tensor_to_image(image.byte())
        else:
            self.image_type = 'numpy'
        self.image = image
        self.visualize = visualize
        if visualize:
            self.original = image
    def postprocess_data(self):
        if self.image_type == 'tensor':
            self.image = kornia.image_to_tensor(self.image, keepdim=False)
        if self.data_type == 'dict':
            data_output = self.data
            data_output['image'] = self.image
        else:
            data_output = self.image

        if self.visualize:
            visualization.plot_image_tranformation({'image':data_output}, {'image':self.original})
        return data_output


class normalization(color_transform):
    def __init___(self, image):
        color_transform.__init__(self, image)

    def __call__(self, *args, **kwargs):
        norm_img = np.zeros((self.image.shape[0], self.image.shape[1]))
        self.image = cv2.normalize(self.image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_img


def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

class brightness(color_transform):
    def __init__(self, image, brightness,  normalized = True):
        color_transform.__init__(self, image)
        #if brightness < 0 or brightness > 2:
        #    raise Exception("Brightness factor value must be between 0 and 2 (Received {}".format(brightness))
        self.image = np.float32(self.image)
        self.brightness =  map(brightness, 0,2, -255, 255)
        #self.brightness = -80
        #self.contrast = constrast
        self.normalized = normalized

    def __call__(self, *args, **kwargs):
        if self.brightness > 0:
            shadow = self.brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + self.brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        self.image = cv2.addWeighted(self.image, alpha_b, self.image, 0, gamma_b)
        return self.postprocess_data()

class contrast(color_transform):
    def __init__(self, image, contrast,  normalized = True):
        color_transform.__init__(self, image)
        self.image = np.float32(self.image)
        self.contrast = map(contrast, 0, 2, -127, 127)
        self.normalized = normalized

    def __call__(self, *args, **kwargs):
        self.image = self.image.copy()
        f = 131 * (self.contrast + 127) / (127 * (131 - self.contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        self.image = cv2.addWeighted(self.image, alpha_c, self.image, 0, gamma_c)

        return self.postprocess_data()

class brigthness_and_contrast(color_transform):
    def __init__(self, image, brightness = 0, contrast = 0):
        color_transform.__init__(self, image)
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, *args, **kwargs):
        if self.brightness != 0:
            if self.brightness > 0:
                shadow = self.brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + self.brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(self.image, alpha_b, self.image, 0, gamma_b)
        else:
            buf = self.image.copy()

        if self.contrast != 0:
            f = 131 * (self.contrast + 127) / (127 * (131 - self.contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        self.image = buf
        return self.postprocess_data()

class gamma(color_transform):
    def __init__(self, image, gamma, visualize):
        color_transform.__init__(self, image, visualize)
        self.gamma = gamma

    def __call__(self, *args, **kwargs):
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, self.gamma) * 255.0, 0, 255)
        self.image = cv2.LUT(self.image, lookUpTable)
        return self.postprocess_data()

class gaussian_noise(color_transform):
    def __init__(self, image, var=0.5,  visualize=False):
        color_transform.__init__(self, image, visualize)
        self.var = var

    def __call__(self, *args, **kwargs):
        self.image = utils._apply_gaussian_noise(self.image, self.var)
        return self.postprocess_data()


class salt_and_peper_noise(color_transform):
    def __init__(self, image, amount, s_vs_p,  visualize=False):
        color_transform.__init__(self, image, visualize)
        self.amount = amount
        self.s_vs_p = s_vs_p

    def __call__(self, *args, **kwargs):
        self.image = utils._apply_salt_and_pepper_noise(self.image, self.amount, self.s_vs_p)
        return self.postprocess_data()


class poisson_noise(color_transform):
    def __init__(self, image,  visualize=False):
        color_transform.__init__(self, image, visualize)

    def __call__(self, *args, **kwargs):
        self.image = utils._apply_poisson_noise(self.image)
        return self.postprocess_data()


class spekle_noise(color_transform):
    def __init__(self, image,  visualize=False):
        color_transform.__init__(self, image, visualize)

    def __call__(self, *args, **kwargs):
        self.image = utils._apply_spekle_noise(self.image)
        return self.postprocess_data()


