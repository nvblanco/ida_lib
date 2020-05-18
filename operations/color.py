from typing import Union

import torch

from . import color_functional
import numpy as np


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


'''GAMA ADJUST: op no lineal para corregir la luminancia de la imagen
    gamma bajo -> img blanca 
    gamma alto -> img oscura
    gamma [0-5]'''

'''

def changue_contrast_and_brightness(data, contrast, brightness, visualize = False):
    op = color_functional.brigthness_and_contrast(data, brightness=brightness,contrast=contrast)
    return op()
    
def normalize(data,visualize=False):
    op = color_functional.normalization(data, visualize)
    return op()

def gamma_adjust(data, gamma=1.8,  visualize = False):
    op = color_functional.gamma(data, gamma, visualize)
    return op()



def bright_lut(data, brigth = 1,  visualize = False):
    op = color_functional.brightness_lut(data, brigth, visualize)
    return op()

##Noise types##
def inyect_gaussian_noise(data, var=0.5, visualize= False):
    op = color_functional.gaussian_noise(data, var, visualize=visualize)
    return op()

def inyect_salt_and_pepper_noise(data, amount = 0.02, s_vs_p = 0.5, visualize= False):
    op = color_functional.salt_and_peper_noise(data, amount, s_vs_p, visualize=visualize)
    return op()

def inyect_poisson_noise(data, visualize = False):
    op = color_functional.poisson_noise(data)
    return op()

def inyect_spekle_noise(data, visualize = False):
    op = color_functional.spekle_noise(data, visualize)
    return op()


def equalize_histogram(data, visualize = False):
    op = color_functional.histogram_equalization(data, visualize)
    return op'''
