from functools import wraps

import cv2
import numpy as np
from . import utils
import torch
import kornia
from image_augmentation import visualization

device = utils.device


def prepare_data_for_opencv(func):

    @wraps(func)
    def wrapped_function(image, visualize,  *args, **kwargs):
        if isinstance(image, dict):
            data_type = 'dict'
            data = image
            image = image['image']
        else:
            data_type = 'image'
        if torch.is_tensor(image):
            image_type = 'tensor'
            image = kornia.tensor_to_image(image.byte())
        else:
            image_type = 'numpy'
        if visualize:
            original = image

        image = func(image, *args, **kwargs) #Execute transform

        if image_type == 'tensor':
            image = kornia.image_to_tensor(image, keepdim=False)
        if data_type == 'dict':
            data_output = data
            data_output['image'] = image
        else:
            data_output = image

        if visualize:
            visualization.plot_image_tranformation({'image': data_output}, {'image': original})
        return data_output

    return wrapped_function

def apply_lut_by_pixel_function(function, image):
    lookUpTable = np.empty((1, 256), np.int16)
    for i in range(256):
        lookUpTable[0, i] = function(i)
    lookUpTable[0, :] = np.clip(lookUpTable[0, :], 0, 255)
    lut =  np.uint8(lookUpTable)
    return cv2.LUT(image, lut)


@prepare_data_for_opencv
def normalize_image(img, norm_type = cv2.NORM_MINMAX):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=norm_type, dtype=cv2.CV_32F)


def get_brigthness_function(brightness):
    return lambda x : x + brightness

@prepare_data_for_opencv
def change_brigthness(image, brightness):
    brightness = utils.map_value(brightness, 0, 2, -256, 256)
    return apply_lut_by_pixel_function(get_brigthness_function(brightness), image)


def get_brigthness_function(brightness):
    return lambda x : x + brightness

@prepare_data_for_opencv
def change_contrast(image, contrast):
    return apply_lut_by_pixel_function(get_contrast_function(contrast), image)

def get_contrast_function(contrast):
    return lambda x : contrast * (x - 255) + 255


@prepare_data_for_opencv
def change_gamma(image, gamma):
    return apply_lut_by_pixel_function(get_gamma_function(gamma), image)

def get_gamma_function(gamma):
    return lambda x : pow(x / 255, gamma) * 255

@prepare_data_for_opencv
def gaussian_noise(image, var = 20):
    return _apply_gaussian_noise(image, var)

@prepare_data_for_opencv
def salt_and_pepper_noise(image, amount , s_vs_p):
    return _apply_salt_and_pepper_noise(image, amount, s_vs_p)

@prepare_data_for_opencv
def poisson_noise(image):
    return _apply_poisson_noise(image)

@prepare_data_for_opencv
def spekle_noise(image, mean=0, var=0.01):
    return _apply_spekle_noise(image, mean,var)


@prepare_data_for_opencv
def histogram_equalization(img):
    for channel in range(img.shape[2]): img[...,channel] = cv2.equalizeHist(img[...,channel])
    return img
    '''ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img'''

@prepare_data_for_opencv
def gaussian_blur(img, blur_size):
    return apply_gaussian_blur(img, blur_size)

@prepare_data_for_opencv
def blur(img, blur_size):
    return apply_gaussian_blur(img, blur_size)



def apply_gaussian_blur(img, blur_size=(5, 5)):
    return cv2.GaussianBlur(img, blur_size,cv2.BORDER_DEFAULT)

def _apply_blur(img,  blur_size=(5, 5)):
    return cv2.blur(img, (5,5))

def _resize_image(image, new_size):
    return cv2.resize(image, new_size)

def _apply_gaussian_noise(image, var = 20):
    gaussian_noise = np.zeros((image.shape[0], image.shape[1],1), dtype=np.uint8)
    cv2.randn(gaussian_noise, 50, 20)
    gaussian_noise = np.concatenate((gaussian_noise, gaussian_noise, gaussian_noise), axis=2)
    gaussian_noise = (gaussian_noise * var).astype(np.uint8)
    return cv2.add(image, gaussian_noise)

def _apply_salt_and_pepper_noise(image, amount=0.05, s_vs_p = 0.5 ):
    if not utils.is_a_normalized_image(image):
        salt = 255
    else:
        salt = 1
    pepper = 0
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords[0], coords[1], :] = salt
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords[0], coords[1], :] = pepper
    return out

def _apply_poisson_noise(image):
    noise = np.random.poisson(40, image.shape)
    return image + noise

def _apply_spekle_noise(image, mean=0, var=0.01):
    gauss = np.random.normal(mean, var ** 0.5, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + image * gauss
    return noisy

'''
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
        return self.postprocess_data()


def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

class brightness(color_transform):
    def __init__(self, image, brightness,  normalized = True):
        color_transform.__init__(self, image)
        #if brightness < 0 or brightness > 2:
        #    raise Exception("Brightness factor value must be between 0 and 2 (Received {}".format(brightness))
        self.image = np.float32(self.image)
        self.brightness =  map(brightness, 0,2, -255, 255)
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

class brightness_lut(color_transform):
    def __init__(self, image, brigthness, visualize):
        color_transform.__init__(self, image, visualize)
        self.brightness = brigthness

    def __call__(self, *args, **kwargs):
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(i + self.brightness, 0, 255)
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
'''
'''class histogram_equalization(color_transform):
    def __init__(self, image, visualize = False):
        color_transform.__init__(self, image, visualize)

    def __call__(self, *args, **kwargs):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        self.image[:, :, 0] = cv2.equalizeHist(self.image[:, :, 0])

        # convert the YUV image back to RGB format
        self.image = cv2.cvtColor(self.image, cv2.COLOR_YUV2BGR)
        return self.postprocess_data()'''


