from typing import Union

from . import geometry_functional
import torch

'''
DATA TYPES:
    * image:    torch tensor (C, H, W)
    * mask:     torch tensor (C, H, W)
    * hetamaps: torch tensor (C, H, W)
    
    * keypoints:list of torch tensor of dims (H, W) --keypoints must be with some 2d data'''

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













