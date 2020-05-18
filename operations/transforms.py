from typing import Union
import numpy as np

import torch

from operations import geometry_functional, color_functional


def hflip(data: dict, visualize : bool = False) -> dict:
    ''' Horizontally flip the input data. '''
    return geometry_functional.hflip_compose_data(data, visualize)

def vflip(data: dict,  visualize : bool = False)-> dict:
    ''' Vertically flip the input data. '''
    return geometry_functional.vflip_compose_data(data, visualize)

def affine(data: dict, matrix: torch.tensor,  visualize : bool = False)-> dict:
    ''' Apply the affine transformation to each element of the input data that represents the input matrix
            matrix: torch tensor 2x3'''
    return geometry_functional.affine_compose_data(data, visualize, matrix)

def rotate(data: dict, degrees: float,  visualize : bool = False, center: Union[None, torch.Tensor] = None)-> dict:
    ''' Rotate each element of the input data by the indicated degrees counterclockwise'''
    return geometry_functional.rotate_compose_data(data, visualize, degrees=degrees, center=center)

def scale(data: dict, scale_factor,  visualize : bool = False, center: Union[None, torch.Tensor] = None)-> dict:
    ''' Scale each element of the input data by the input factor.
        * scale_factor < 1 -> output image is smaller than input one
        * scale_factor = 1 -> output image is is the same as the input image
        * scale_factor = 2 -> each original pixel occupies 2 pixels in the output image

        If center is None, it is taken the center of the image to apply the scale'''
    return geometry_functional.scale_compose_data(data, visualize, center=center, scale_factor=scale_factor)

def translate(data: dict, translation: tuple,  visualize : bool = False)-> dict:
    ''' Translate input by the input translation.
            Translation: 2-tuple (axisX pixels translation, axixY pixels translation)'''
    return geometry_functional.translate_compose_data(data, visualize, translation)

def shear(data: dict, shear_factor: tuple,  visualize : bool = False)-> dict:
    ''' Shear input data by the input shear factor
            shear_factor: 2-tuple (axisX shearFactor, axisY shearFactor)
                shear_factor = (0,0) -> output image is is the same as the input image
                0 > shear_factor > 1 -> recommended values to avoid excessive deformation'''
    return geometry_functional.shear_compose_data(data, visualize, shear_factor)

def change_contrast(data: Union[dict, torch.Tensor, np.ndarray], contrast: float, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' change the image contrast. if the input data is a dictionary, only those corresponding to an image are altered
            * contrast = 0 -> removes image constrast (white output image)
            * contrast = 1 -> remains unchanged
            * contrast > 1 -> increases contrast'''
    return color_functional.change_contrast(data, visualize=visualize, contrast=contrast)


def change_brigntness(data: Union[dict, torch.Tensor, np.ndarray], brigth=1, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' change the image brigthness. if the input data is a dictionary, only those corresponding to an image are altered
             * brigthness = 0 -> removes image brigthness (balck output image)
             * brigthness = 1 -> remains unchanged
             * brigthness > 1 -> increases brigthness'''
    return color_functional.change_brigthness(data, brightness=brigth, visualize=visualize)


def change_gamma(data: Union[dict, torch.Tensor, np.ndarray], gamma, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' adjust image's gamma (luminance correction) . if the input data is a dictionary, only those corresponding to an image are altered
              * gamma = 0 -> removes image brigthness (balck output image)
              * gamma = 1 -> remains unchanged
              * gamma > 1 -> increases brigthness'''
    return color_functional.change_gamma(data, gamma=gamma, visualize=visualize)


def equalize_histogram(data: Union[dict, torch.Tensor, np.ndarray], visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' equalize image histogram. if the input data is a dictionary, only those corresponding to an image are altered'''
    return color_functional.histogram_equalization(data, visualize=visualize)


def inyect_gaussian_noise(data: Union[dict, torch.Tensor, np.ndarray], var=0.5, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' inyect gaussian noise. If the input data is a dictionary, only those corresponding to an image are altered
                 * var : var of the gaussian distribution of noise'''
    return color_functional.gaussian_noise(data, var=var, visualize=visualize)


def inyect_salt_and_pepper_noise(data: Union[dict, torch.Tensor, np.ndarray], amount=0.05, s_vs_p=0.5, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' Inyect salt and pepper noisse if the input data is a dictionary, only those corresponding to an image are altered
             * amount: percentage of image's pixels to be occupied by noise
             * s_vs_p: noise type distribution. Default same salt (white pixel) as pepper (black pixels)'''
    return color_functional.salt_and_pepper_noise(data, amount=amount, s_vs_p=s_vs_p, visualize=visualize)


def inyect_poisson_noise(data: Union[dict, torch.Tensor, np.ndarray], visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' Inyect poisson noisse. if the input data is a dictionary, only those corresponding to an image are altered'''
    return color_functional.poisson_noise(data, visualize=visualize)


def inyect_spekle_noise(data: Union[dict, torch.Tensor, np.ndarray], mean=0, var=0.01, visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' Inyect poisson noisse. if the input data is a dictionary, only those corresponding to an image are altered'''
    return color_functional.spekle_noise(data, mean=mean, var=var, visualize=visualize)


def gaussian_blur(data: Union[dict, torch.Tensor, np.ndarray], blur_size=(5, 5), visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' Blurring an image by a Gaussian function.  if the input data is a dictionary, only those corresponding to an image are altered'''
    return color_functional.gaussian_blur(data, blur_size=blur_size, visualize=visualize)


def blur(data: Union[dict, torch.Tensor, np.ndarray], blur_size=(5, 5), visualize: bool = False) -> Union[
    dict, torch.Tensor, np.ndarray]:
    ''' Blur image.  if the input data is a dictionary, only those corresponding to an image are altered'''
    return color_functional.blur(data, blur_size=blur_size, visualize=visualize)