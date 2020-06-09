from typing import Union

import numpy as np
import torch

from ida_lib.operations import geometry_ops_functional, pixel_ops_functional

__all__ = ['hflip',
           'vflip',
           'affine',
           'rotate',
           'shear',
           'scale',
           'translate',
           'change_gamma',
           'change_contrast',
           'change_brigntness',
           'equalize_histogram',
           'inyect_gaussian_noise',
           'inyect_poisson_noise',
           'inyect_spekle_noise',
           'inyect_salt_and_pepper_noise',
           'blur',
           'gaussian_blur']

def hflip(data: dict, visualize : bool = False) -> dict:
    """
     Horizontally flip the input data.
    :param data : dict of elements to be transformed
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return geometry_ops_functional.hflip_compose_data(data, visualize)

def vflip(data: dict,  visualize : bool = False)-> dict:
    """
    Vertically flip the input data.
    :param data : dict of elements to be transformed
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return geometry_ops_functional.vflip_compose_data(data, visualize)

def affine(data: dict, matrix: torch.tensor,  visualize : bool = False)-> dict:
    """
    Apllies affine transformation to the data
    :param data : dict of elements to be transformed
    :param matrix : matrix of transformation
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return geometry_ops_functional.affine_compose_data(data, visualize, matrix)

def rotate(data: dict, degrees: float,  visualize : bool = False, center: Union[None, torch.Tensor] = None)-> dict:
    """
    Rotate each element of the input data by the indicated degrees counterclockwise
    :param data : dict of elements to be transformed
    :param degrees : degrees of rotation
    :param center : center of rotation. If center is None, it is taken the center of the image to apply the rotation
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return geometry_ops_functional.rotate_compose_data(data, visualize, degrees=degrees, center=center)

def scale(data: dict, scale_factor,  visualize : bool = False, center: Union[None, torch.Tensor] = None)-> dict:
    """
    Scale each element of the input data by the input factor.
    :param data : dict of elements to be transformed
    :param scale_factor: fcator of scaling to be applied
        * scale_factor < 1 -> output image is smaller than input one
        * scale_factor = 1 -> output image is is the same as the input image
        * scale_factor = 2 -> each original pixel occupies 2 pixels in the output image
    :param center : center of scaling. If center is None, it is taken the center of the image to apply the scale
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return geometry_ops_functional.scale_compose_data(data, visualize, center=center, scale_factor=scale_factor)

def translate(data: dict, translation: tuple,  visualize : bool = False)-> dict:
    """
    Translate input by the input translation.
    :param data : dict of elements to be transformed
    :param translation  : number of pixels to be translated
    :param visualize  : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return geometry_ops_functional.translate_compose_data(data, visualize, translation)

def shear(data: dict, shear_factor: tuple,  visualize : bool = False)-> dict:
    """
    Shear input data by the input shear factor
    :param data : dict of elements to be transformed
    :param shear_factor : pixels to shear
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return geometry_ops_functional.shear_compose_data(data, visualize, shear_factor)

def change_contrast(data: Union[dict, torch.Tensor, np.ndarray], contrast: float, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
     change the image contrast. if the input data is a dictionary, only those corresponding to an image are altered
    :param data : dict of elements to be transformed
    :param contrast: desired constrast factor to the data
            * contrast = 0 -> removes image constrast (white output image)
            * contrast = 1 -> remains unchanged
            * contrast > 1 -> increases contrast
    :param visualize  : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.change_contrast(data, visualize=visualize, contrast=contrast)


def change_brigntness(data: Union[dict, torch.Tensor, np.ndarray], brigth=1, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
    change the image brigthness. if the input data is a dictionary, only those corresponding to an image are altered
    :param data : dict of elements to be transformed
    :param brigth : desired brigthness amount for the data
             * brigthness = 0 -> removes image brigthness (balck output image)
             * brigthness = 1 -> remains unchanged
             * brigthness > 1 -> increases brigthnes
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.change_brigthness(data, brightness=brigth, visualize=visualize)


def change_gamma(data: Union[dict, torch.Tensor, np.ndarray], gamma, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
    adjust image's gamma (luminance correction) . if the input data is a dictionary, only those corresponding to an image are altered
    :param data : dict of elements to be transformed
    :param gamma  : desired gamma factor (luminance of image)
              * gamma = 0 -> removes image luminance (balck output image)
              * gamma = 1 -> remains unchanged
              * gamma > 1 -> increases luminance
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.change_gamma(data, gamma=gamma, visualize=visualize)


def equalize_histogram(data: Union[dict, torch.Tensor, np.ndarray], visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
    equalize image histogram. if the input data is a dictionary, only those corresponding to an image are altered
    :param data : dict of elements to be transformed
    :param visualize: if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.histogram_equalization(data, visualize=visualize)


def inyect_gaussian_noise(data: Union[dict, torch.Tensor, np.ndarray], var=0.5, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
    inyect gaussian noise. If the input data is a dictionary, only those corresponding to an image are altered
    :param data: dict of elements to be transformed
    :param var: varianze of the noise distribution
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.gaussian_noise(data, var=var, visualize=visualize)


def inyect_salt_and_pepper_noise(data: Union[dict, torch.Tensor, np.ndarray], amount=0.05, s_vs_p=0.5, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
    Inyect salt and pepper noisse if the input data is a dictionary, only those corresponding to an image are altered
    :param data : dict of elements to be transformed
    :param amount : percentage of image's pixels to be occupied by noise
    :param s_vs_p  : noise type distribution. Default same salt (white pixel) as pepper (black pixels)
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.salt_and_pepper_noise(data, amount=amount, s_vs_p=s_vs_p, visualize=visualize)


def inyect_poisson_noise(data: Union[dict, torch.Tensor, np.ndarray], visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
    Inyect poisson noisse. if the input data is a dictionary, only those corresponding to an image are altered
    :param data : dict of elements to be transformed
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.poisson_noise(data, visualize=visualize)


def inyect_spekle_noise(data: Union[dict, torch.Tensor, np.ndarray], mean=0, var=0.01, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
    Inyect poisson noisse. if the input data is a dictionary, only those corresponding to an image are altered
    :param data : dict of elements to be transformed
    :param mean : mean of noise distribution
    :param var  : varianze of noise distribution
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.spekle_noise(data, mean=mean, var=var, visualize=visualize)


def gaussian_blur(data: Union[dict, torch.Tensor, np.ndarray], blur_size=(5, 5), visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
    Blurring an image by a Gaussian function.  if the input data is a dictionary, only those corresponding to an image are altered
    :param data : dict of elements to be transformed
    :param blur_size : number of surrounding pixels affecting each output pixel. (pixels on axis X, pixels on axis y)
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.gaussian_blur(data, blur_size=blur_size, visualize=visualize)


def blur(data: Union[dict, torch.Tensor, np.ndarray], blur_size=(5, 5), visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    """
    Blur image.  if the input data is a dictionary, only those corresponding to an image are altered
    :param data : dict of elements to be transformed
    :param blur_size  : number of surrounding pixels affecting each output pixel. (pixels on axis X, pixels on axis y)
    :param visualize : if true it activates the display tool to debug the transformation
    :return: transformed data
    """
    return pixel_ops_functional.blur(data, blur_size=blur_size, visualize=visualize)