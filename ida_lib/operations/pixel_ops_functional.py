from functools import wraps
from typing import Union

import cv2
import kornia
import numpy as np
import torch

from . import utils
from .utils import add_new_axis
from ..visualization import plot_image_transformation


def prepare_data_for_opencv(func):
    """
    Decorator that prepares the input data to apply the transformation that only affects the image (color).
    :param func: color function to be applied to the data
    :return: processed data
    """

    @wraps(func)
    def wrapped_function(image: Union[dict, torch.tensor, np.ndarray], visualize: bool, *args, **kwargs) -> Union[
        dict, torch.tensor, np.ndarray]:

        data = None
        original = None

        if isinstance(image, dict):
            if 'image' not in image:
                raise AttributeError(
                    'it is necessary that the input dictum has some image type element for this type of operation')
            data_type = 'dict'
            data = image
            image = image['image']
        else:
            data_type = 'image'
        if torch.is_tensor(image):
            image_type = 'tensor'
            original_type = image.dtype
            image = kornia.tensor_to_image(image.byte())
        else:
            image_type = 'numpy'
            original_type = image.dtype
        if visualize:
            original = image.copy()

        image = func(image, *args, **kwargs)  # Execute transform

        if len(image.shape) < 3:
            image = image[..., np.newaxis]
        if image_type == 'tensor':
            image = kornia.image_to_tensor(image, keepdim=False)
            image = image.type(original_type)
        else:
            image = image.astype(original_type)
        if data_type == 'dict':
            data_output = data
            data_output['image'] = image
        else:
            data_output = image
        if visualize:
            plot_image_transformation({'image': data_output['image']}, {'image': original})
        return data_output

    return wrapped_function


def apply_lut_by_pixel_function(function, image: np.ndarray) -> np.ndarray:
    """
    Applies the input operation to the image using a LUT

    :param function: Mathematical function that represents the operation to carry out in each pixel of the image
    :param image: input image
    :return:
    """
    look_up_table = np.empty((1, 256), np.int16)
    for i in range(256):
        look_up_table[0, i] = function(i)
    look_up_table[0, :] = np.clip(look_up_table[0, :], 0, 255)
    lut = np.uint8(look_up_table)
    return cv2.LUT(image, lut)


@prepare_data_for_opencv
def normalize_image(img: np.ndarray, norm_type: int = cv2.NORM_MINMAX) -> np.ndarray:
    """
    Normalize the input image

    :param img: input image to be normalized
    :param norm_type: opencv normalization type (' cv2.NORM_MINMAX' |cv2.NORM_HAMMING |cv2.NORM_HAMMING2
                        |cv2.NORM_INF |cv2.NORM_RELATIVE ...)
    :return: normalized image
    """
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=norm_type, dtype=cv2.CV_32F)


def get_brightness_function(brightness: int):
    """

    :param brightness: brightness factor
    :return:    Return the lambda function of the brightness operation
    """
    return lambda x: x + brightness


@prepare_data_for_opencv
def change_brightness(image: Union[dict, torch.tensor, np.ndarray], brightness: int) -> Union[
    dict, torch.tensor, np.ndarray]:
    """
    Change the brightness of the input image.

    :param image: input image to be normalized
    :param brightness: desired amount of brightness for the image
                 0 - no brightness
                 1 - same
                 2 - max brightness
    :return: transformed image
    """
    original_type = image.dtype
    brightness = utils.map_value(brightness, 0, 2, -256, 256)
    return apply_lut_by_pixel_function(get_brightness_function(brightness), image.astype(np.uint8)).astype(
        original_type)


@prepare_data_for_opencv
def change_contrast(image: Union[dict, torch.tensor, np.ndarray], contrast) -> Union[dict, torch.tensor, np.ndarray]:
    """

    :param image  : input image to be transformed
    :param contrast: modification factor to be applied to the image contrast
            * 0  - total contrast removal
            * 1  - dont modify
            * >1 - augment contrast
    :return: returns the transformed image
    """
    original_type = image.dtype
    return apply_lut_by_pixel_function(get_contrast_function(contrast), image.astype(np.uint8)).astype(original_type)


def get_contrast_function(contrast: float):
    """

    :param contrast: modification factor to be applied to the image contrast
    :return: Return the lambda function of the contrast operation
    """
    return lambda x: contrast * (x - 255) + 255


@prepare_data_for_opencv
def change_gamma(image: Union[dict, torch.tensor, np.ndarray], gamma: float) -> Union[dict, torch.tensor, np.ndarray]:
    """

    :param image: input image to be transformed
    :param gamma: desired factor gamma
            * gamma = 0 -> removes image luminance (black output image)
            * gamma = 1 -> remains unchanged
            * gamma > 1 -> increases luminance
    :return: returns the transformed image
    """
    original_type = image.dtype
    return apply_lut_by_pixel_function(get_gamma_function(gamma), image.astype(np.uint8)).astype(original_type)


def get_gamma_function(gamma):
    """

    :param gamma: desired factor gamma
    :return: Returns the lambda function of the gamma adjust operation
    """
    return lambda x: pow(x / 255, gamma) * 255


@prepare_data_for_opencv
def gaussian_noise(image: Union[dict, torch.tensor, np.ndarray], var=20) -> Union[dict, torch.tensor, np.ndarray]:
    """

    :param image: input image to be transformed
    :param var: var of the gaussian distribution of noise
    :return: returns the transformed image
    """
    original_type = image.dtype
    return apply_gaussian_noise(image.astype(np.uint8), var).astype(original_type)


@prepare_data_for_opencv
def salt_and_pepper_noise(image: Union[dict, torch.tensor, np.ndarray], amount, s_vs_p) -> Union[
    dict, torch.tensor, np.ndarray]:
    """

    :param image: input image to be transformed
    :param amount: percentage of image's pixels to be occupied by noise
    :param s_vs_p: percentage of salt respect total noise. Default same salt (white pixel) as pepper (black pixels)
    :return: returns the transformed image
    """
    return apply_salt_and_pepper_noise(image, amount, s_vs_p)


@prepare_data_for_opencv
def poisson_noise(image: Union[dict, torch.tensor, np.ndarray]) -> Union[dict, torch.tensor, np.ndarray]:
    """

    :param image: input image to be transformed
    :return: returns the transformed image
    """
    original_type = image.dtype
    return (apply_poisson_noise(image)).astype(original_type)


@prepare_data_for_opencv
def spekle_noise(image: Union[dict, torch.tensor, np.ndarray], mean=0, var=0.01) -> Union[
    dict, torch.tensor, np.ndarray]:
    """

    :param image: input image to be transformed
    :param mean: mean of noise distribution
    :param var: variance of noise distribution
    :return: returns the transformed image
    """
    original_type = image.dtype
    return (apply_spekle_noise(image, mean, var)).astype(original_type)


@prepare_data_for_opencv
def histogram_equalization(img: Union[dict, torch.tensor, np.ndarray]) -> Union[dict, torch.tensor, np.ndarray]:
    """

    :param img: input image to be transformed
    :return: returns the transformed image
    """
    for channel in range(img.shape[2]):
        img[..., channel] = cv2.equalizeHist(img[..., channel].astype(np.uint8))
    return img


@prepare_data_for_opencv
def gaussian_blur(img: Union[dict, torch.tensor, np.ndarray], blur_size) -> Union[dict, torch.tensor, np.ndarray]:
    """

    :param img: input image to be transformed
    :param blur_size: number of surrounding pixels affecting each output pixel. (pixels on axis X, pixels on axis y)
    :return: returns the transformed image
    """
    return apply_gaussian_blur(img, blur_size)


@prepare_data_for_opencv
def blur(img: Union[dict, torch.tensor, np.ndarray], blur_size) -> Union[dict, torch.tensor, np.ndarray]:
    """

    :param img: input image to be transformed
    :param blur_size: number of surrounding pixels affecting each output pixel. (pixels on axis X, pixels on axis y)
    :return: returns the transformed image
    """
    return apply_gaussian_blur(img, blur_size)


def apply_gaussian_blur(img, blur_size=(5, 5)):
    """

    :param img: input image to be transformed
    :param blur_size: number of surrounding pixels affecting each output pixel. (pixels on axis X, pixels on axis y)
    :return: returns the transformed image
    """
    return cv2.GaussianBlur(img, blur_size, cv2.BORDER_DEFAULT)


def apply_blur(img, blur_size=(5, 5)):
    return cv2.blur(img, blur_size)


def _resize_image(image, new_size):
    return cv2.resize(image, new_size)


def apply_gaussian_noise(image, var=20):
    g_noise = np.zeros((image.shape[0], image.shape[1], 1), dtype=image.dtype)
    cv2.randn(g_noise, 50, 20)
    if len(image.shape) == 3 and image.shape[2] != 1:
        g_noise = np.concatenate((g_noise, g_noise, g_noise), axis=2)
    g_noise = (g_noise * var).astype(image.dtype)
    return cv2.add(image, g_noise)


def apply_salt_and_pepper_noise(image, amount=0.05, s_vs_p=0.5):
    if len(image.shape) == 2:
        image = add_new_axis(image)
    if not utils.is_a_normalized_image(image):
        salt = 255
    else:
        salt = 1
    pepper = 0
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape if i != 1]
    out[coords[0], coords[1], :] = salt
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape if i != 1]
    out[coords[0], coords[1], :] = pepper
    return out


def apply_poisson_noise(image):
    noise = np.random.poisson(40, image.shape)
    return image + noise


def apply_spekle_noise(image, mean=0, var=0.01):
    gauss = np.random.normal(mean, var ** 0.5, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + image * gauss
    return noisy
