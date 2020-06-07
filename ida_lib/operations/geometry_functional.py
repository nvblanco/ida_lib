from functools import wraps
from string import digits
from typing import Union

import kornia
import torch

from ida_lib.image_augmentation import visualization
from . import utils

device = utils.device
data_types_2d = {"image", "mask", "heatmap"}
data_types_1d = {"keypoints"}

one_torch = torch.ones(1).to(device)


def prepare_data(func):
    """
    Decorator that prepares the input data to apply the geometric transformation. For this purpose, it concatenates all
    the two-dimensional elements of the input data in the same tensor on which a single transformation is applied.  If
    the input data contains point coordinates, they are grouped in a matrix as homogeneous coordinates, over which a
    single matrix multiplication is performed.

    :param func: geometric function to be applied to the data
    :return: processed data
    """

    @wraps(func)
    def wrapped_function(data: dict, visualize: bool,  *args, **kwargs):
        process_data = {}
        process_data['points'] = None
        if visualize:
            process_data['original'] = data
        if isinstance(data, dict):
            process_data['types_2d'] = {}
            compose_data = torch.tensor([])
            remove_digits = str.maketrans('', '', digits)
            for type in data.keys():
                no_numbered_type = type.translate(remove_digits)
                if no_numbered_type in data_types_2d:
                    if not type in data_types_2d: data_types_2d.add(
                        type)
                    compose_data = torch.cat((compose_data, data[type]),
                                             0)  # concatenate data into one multichannel pytoch tensor
                    process_data['types_2d'][type] = data[type].shape[0]
                else:
                    process_data['points_matrix'] = data[type]
            process_data['data2d'] = compose_data.to(device)
            if process_data['points_matrix'] is not None: process_data['points_matrix'] = utils.keypoints_to_homogeneus_and_concatenate(
                process_data['points_matrix'] )
        else:
            if data.dim() < 3:
                raise Exception("Single data must be al least 3 dims")
            else:
                process_data['points_matrix'] = None
                process_data['types_2d'] = {}
                process_data['types_2d']['image'] = data.shape[0]
        data_output = func(process_data, *args, **kwargs) #Execute transform

        data_output['data2d'] = data_output['data2d'].cpu()
        if 'types_2d' in  data_output:
            data_process = {}
            data_split = torch.split(data_output['data2d'], list(data_output['types_2d'].values()), dim=0)
            for index, type in enumerate(data_output['types_2d']):
                data_process[type] = data_split[index]
            if 'mask' in data_process: data_process['mask'] = utils.mask_change_to_01_functional(
                data_process['mask'])
            if 'points_matrix' in data_output:
                data_process['keypoints'] = [
                ((dato.cpu())[:2, :]).reshape(2) for dato in torch.split(data_output['points_matrix'], 1, dim=1)]
        else:
            data_process = data_output['data2d']
        if visualize:
            visualization.plot_image_tranformation(data_process, data_output['original'])
        return data_process

    return wrapped_function


"""---Vertical Flip---"""
def vflip_image(img: torch.tensor)-> torch.tensor:
    return kornia.vflip(img)

def vflip_coordiantes_matrix(matrix: torch.tensor, heigth: int)-> torch.tensor:
    matrix[1] = torch.ones(1, matrix.shape[1]).to(device) * (heigth) - \
    matrix[1]
    return matrix

@prepare_data
def vflip_compose_data(data: dict)->dict:
    """
    :param data : dict of elements to be transformed
    :return: transformed data
    """
    data['data2d'] = vflip_image(data['data2d'])
    heigth = data['data2d'].shape[-2]
    if 'mask'in data:
        data['points_matrix'] = vflip_coordiantes_matrix(data['points_matrix'], heigth)
    return data

"""--- Horizontal Flip---"""

def hflip_image(img: torch.tensor)-> torch.tensor:
    return kornia.hflip(img)

def hflip_coordinates_matrix(matrix: torch.tensor, width: int)-> torch.tensor:
    matrix[0] = torch.ones(1, matrix.shape[1]).to(device) * (width) - \
                               matrix[0]
    return matrix

@prepare_data
def hflip_compose_data(data: dict) -> dict:
    """
    :param data : dict of elements to be transformed
    :return: transformed data
    """
    data['data2d'] = hflip_image(data['data2d'])
    width = data['data2d'].shape[-1]
    if 'points_matrix' in data:
        data['points_matrix'] = hflip_coordinates_matrix(data['points_matrix'], width)
    return data

""" --- Afine transform ---"""

def affine_image(img: torch.tensor, matrix: torch.tensor)-> torch.tensor:
    return  kornia.geometry.affine(img, matrix)

def affine_coordinates_matrix(matrix_coordinates: torch.tensor, matrix_transformation: torch.tensor) -> torch.tensor:
    return torch.matmul(matrix_transformation, matrix_coordinates)

@prepare_data
def affine_compose_data(data: dict, matrix: torch.tensor) -> dict:
    """
    :param data : dict of elements to be transformed
    :param matrix : matrix of transformation
    :return: transformed data
    """
    matrix = matrix.to(device)
    data['data2d'] = affine_image(data['data2d'], matrix)
    if 'points_matrix' in data:
        data['points_matrix'] = affine_coordinates_matrix(data['points_matrix'], matrix)
    return data

""" --- Rotate transform --- """
def get_rotation_matrix(center: torch.tensor, degrees: torch.tensor):
    return ( kornia.geometry.get_rotation_matrix2d(angle=degrees, center=center, scale=one_torch)).reshape(2, 3)

def rotate_image(img: torch.tensor, degrees: torch.tensor, center: torch.tensor)-> torch.tensor:
    return  kornia.geometry.rotate(img, angle=degrees, center=center)

def rotate_coordinates_matrix(matrix_coordinates: torch.tensor, matrix: torch.tensor)-> torch.tensor:
    return torch.matmul(matrix, matrix_coordinates)

@prepare_data
def rotate_compose_data(data: dict, degrees: torch.tensor, center: torch.tensor):
    """
    :param data : dict of elements to be transformed
    :param degrees : counterclockwise degrees of rotation
    :param center : center of rotation. Default, center of the image
    :return: transformed data
    """
    degrees = degrees * one_torch
    if center is None:
        center = utils.get_torch_image_center(data['data2d'])
    else:
        center = center
    center = center.to(device)
    data['data2d'] = rotate_image(data['data2d'], degrees, center)
    matrix = get_rotation_matrix(center, degrees)
    if 'points_matrix' in data:
        data['points_matrix'] = rotate_coordinates_matrix(data['points_matrix'], matrix)
    return data

""" ---Scale Transform----"""
def get_scale_matrix(center: torch.tensor, scale_factor: Union[float, torch.tensor]):
    if isinstance(scale_factor,
                  float) or scale_factor.dim() == 1:  # si solo se proporciona un valor; se escala por igual en ambos ejes
        scale_factor = torch.ones(2).to(device) * scale_factor
    matrix = torch.zeros(2, 3).to(device)
    matrix[0, 0] = scale_factor[0]
    matrix[1, 1] = scale_factor[1]
    matrix[0, 2] = (-scale_factor[0] + 1) * center[:, 0]
    matrix[1, 2] = (-scale_factor[1] + 1) * center[:, 1]
    return matrix

def scale_image(img: torch.tensor, scale_factor: torch.tensor, center: torch.tensor) -> torch.tensor:
    return  kornia.geometry.scale(img, scale_factor=scale_factor, center=center)

def scale_coordinates_matrix(matrix_coordinates: torch.tensor, matrix: torch.tensor) -> torch.tensor:
    return torch.matmul(matrix, matrix_coordinates)

@prepare_data
def scale_compose_data(data: dict, scale_factor: Union[float, torch.tensor], center: Union[torch.tensor, None]=None) -> dict:
    """
    :param data: dict of elements to be transformed
    :param scale_factor  : factor of scaling
    :param center : center of scaling. By default its taken the center of the image
    :return: transformed data
    """
    scale_factor = (torch.ones(1) * scale_factor).to(device)
    if center is None:
        center = utils.get_torch_image_center(data['data2d'])
    center = center.to(device)
    data['data2d'] = scale_image(data['data2d'], scale_factor, center)
    matrix = get_scale_matrix(center, scale_factor)
    if 'points_matrix' in data:
        data['points_matrix'] = scale_coordinates_matrix( data['points_matrix'], matrix)
    return data

""" --- Translation transform ---"""
def translate_image(img: torch.tensor, translation: torch.tensor)-> torch.tensor:
    return kornia.geometry.translate(img, translation)

def translate_coordinates_matrix(matrix_coordinates: torch.tensor, translation: torch.tensor) -> torch.tensor:
    matrix = torch.zeros((3, matrix_coordinates.shape[1])).to(device)
    row = torch.ones((1, matrix_coordinates.shape[1])).to(device)
    matrix[0] = row * translation[:, 0]
    matrix[1] = row * translation[:, 1]
    return  matrix_coordinates + matrix

@prepare_data
def translate_compose_data(data: dict, translation: Union[int, torch.tensor]) -> dict:
    """
    :param data : dict of elements to be transformed
    :param translation : number of pixels to translate
    :return: transformed data
    """
    if not torch.is_tensor(translation):
        translation = (torch.tensor(translation).float().reshape((1, 2)))
    translation = translation.to(device)
    data['data2d'] = translate_image(data['data2d'], translation)
    if 'points_matrix' in data:
        data['points_matrix'] =  translate_coordinates_matrix(data['points_matrix'], translation)
    return data


""" --- Shear Transform ---"""
def get_shear_matrix(shear_factor: torch.tensor) -> torch.tensor:
    matrix = torch.eye(2, 3).to(device)
    matrix[0, 1] = shear_factor[...,0]
    matrix[1, 0] = shear_factor[...,1]
    return matrix

def shear_image(img: torch.tensor, shear_factor: torch.tensor) -> torch.tensor:
    return kornia.geometry.affine(img, shear_factor)

def shear_coordinates_matrix(matrix_coordinates: torch.tensor, matrix: torch.tensor) -> torch.tensor:
    return  torch.matmul(matrix, matrix_coordinates)

@prepare_data
def shear_compose_data(data: dict, shear_factor: Union[float, torch.tensor]) -> dict:
    """
    :param data : dict of elements to be transformed
    :param shear_factor : pixels of shearing
    :return:
    """
    shear_factor = (torch.tensor(shear_factor).reshape(1,2)).to(device)
    matrix = get_shear_matrix(shear_factor)
    data['data2d'] = shear_image(data['data2d'], matrix)
    matrix = get_shear_matrix(shear_factor)
    if 'points_matrix' in data:
        data['points_matrix'] =  shear_coordinates_matrix(data['points_matrix'], matrix)
    return data

