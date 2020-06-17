from functools import wraps
from typing import Union

import kornia
import torch

from ida_lib.global_parameters import device, one_torch, identity
from . import utils
from .utils import data_to_numpy, get_principal_type, dtype_to_torch_type, is_numpy_data
from ..core.pipeline_functional import preprocess_data, postprocess_data
from ..visualization import plot_image_transformation


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
    def wrapped_function(data: dict, visualize: bool, interpolation: str = 'bilinear', *args, **kwargs):
        principal_type = get_principal_type(data)
        is_numpy = is_numpy_data(data)
        data = data_to_numpy(data)
        original_type = dtype_to_torch_type(data[principal_type].dtype)
        data_original = data.copy()
        p_data, data_info = preprocess_data(data=data, interpolation=interpolation)
        output = func(p_data, *args, **kwargs)  # Execute transform
        output = postprocess_data(batch=[output], batch_info=data_info, data_original=None,
                                  original_type=original_type)
        if visualize:
            plot_image_transformation(output, data_original)
        if is_numpy:
            output = data_to_numpy(output)
        return output

    return wrapped_function


"""---Vertical Flip---"""


def vflip_image(img: torch.tensor) -> torch.tensor:
    return kornia.vflip(img)


def vflip_coordinates_matrix(matrix: torch.tensor, height: int) -> torch.tensor:
    matrix[1] = torch.ones(1, matrix.shape[1]).to(device) * height - \
                matrix[1]
    return matrix


@prepare_data
def vflip_compose_data(data: dict) -> dict:
    """

    :param data: dict of elements to be transformed
    :return: transformed data
    """
    if len(data['data_2d']) != 0:
        data['data_2d'] = vflip_image(data['data_2d'])
        height = data['data_2d'].shape[-2]
    else:
        height = data['data_2d_discrete'].shape[-2]
    if 'data_2d_discrete' in data:
        data['data_2d_discrete'] = vflip_image(data['data_2d_discrete'])
    if 'mask' in data:
        data['points_matrix'] = vflip_coordinates_matrix(data['points_matrix'], height)
    return data


"""--- Horizontal Flip---"""


def hflip_image(img: torch.tensor) -> torch.tensor:
    return kornia.hflip(img)


def hflip_coordinates_matrix(matrix: torch.tensor, width: int) -> torch.tensor:
    matrix[0] = torch.ones(1, matrix.shape[1]).to(device) * width - \
                matrix[0]
    return matrix


@prepare_data
def hflip_compose_data(data: dict) -> dict:
    """

    :param data: dict of elements to be transformed
    :return: transformed data
    """
    if len(data['data_2d']) != 0:
        data['data_2d'] = hflip_image(data['data_2d'])
        width = data['data_2d'].shape[-1]
    else:
        width = data['data_2d_discrete'].shape[-1]
    if 'data_2d_discrete' in data:
        data['data_2d_discrete'] = hflip_image(data['data_2d_discrete'])

    if 'points_matrix' in data:
        data['points_matrix'] = hflip_coordinates_matrix(data['points_matrix'], width)
    return data


""" --- Affine transform ---"""


def affine_image(img: torch.tensor, matrix: torch.tensor, interpolation: str = 'bilinear',
                 padding_mode: str = 'zeros') -> torch.tensor:
    if matrix.shape[0] == 3:
        matrix = matrix[:2, :]
    return own_affine(img, matrix, interpolation=interpolation, padding_mode=padding_mode)


def affine_coordinates_matrix(matrix_coordinates: torch.tensor, matrix_transformation: torch.tensor) -> torch.tensor:
    return torch.matmul(matrix_transformation, matrix_coordinates)


@prepare_data
def affine_compose_data(data: dict, matrix: torch.tensor) -> dict:
    """

    :param data: dict of elements to be transformed
    :param matrix: matrix of transformation
    :return: transformed data
    """
    matrix = matrix.to(device)
    if len(data['data_2d']) != 0:
        data['data_2d'] = affine_image(data['data_2d'], matrix)
    if 'data_2d_discrete' in data:
        data['data_2d_discrete'] = affine_image(data['data_2d_discrete'], matrix, interpolation='nearest')
    if 'points_matrix' in data:
        data['points_matrix'] = affine_coordinates_matrix(data['points_matrix'], matrix)
    return data


""" --- Rotate transform --- """


def get_rotation_matrix(center: torch.tensor, degrees: torch.tensor):
    return (kornia.geometry.get_rotation_matrix2d(angle=degrees, center=center, scale=one_torch)).reshape(2, 3)


def rotate_image(img: torch.tensor, degrees: torch.tensor, center: torch.tensor) -> torch.tensor:
    """mode"""
    return kornia.geometry.rotate(img, angle=degrees, center=center)


def rotate_coordinates_matrix(matrix_coordinates: torch.tensor, matrix: torch.tensor) -> torch.tensor:
    return torch.matmul(matrix, matrix_coordinates)


@prepare_data
def rotate_compose_data(data: dict, degrees: torch.tensor, center: torch.tensor):
    """

    :param data: dict of elements to be transformed
    :param degrees: counterclockwise degrees of rotation
    :param center: center of rotation. Default, center of the image
    :return: transformed data
    """
    ppal = 'data_2d'
    if len(data['data_2d']) == 0:
        ppal = 'data_2d_discrete'
    degrees = degrees * one_torch
    if center is None:
        center = utils.get_torch_image_center(data[ppal])
    else:
        center = center
    center = center.to(device)
    matrix = get_rotation_matrix(center, degrees)
    if len(data['data_2d']) != 0:
        data['data_2d'] = affine_image(data['data_2d'], matrix)
    if 'data_2d_discrete' in data:
        data['data_2d_discrete'] = affine_image(data['data_2d_discrete'], matrix, interpolation='nearest')
    if 'points_matrix' in data:
        data['points_matrix'] = rotate_coordinates_matrix(data['points_matrix'], matrix)
    return data


""" ---Scale Transform----"""


def config_scale_matrix(scale_factor, center, matrix):
    matrix[0, 0] = scale_factor[0]
    matrix[1, 1] = scale_factor[1]
    matrix[0, 2] = (-scale_factor[0] + 1) * center[:, 0]
    matrix[1, 2] = (-scale_factor[1] + 1) * center[:, 1]
    return matrix


def get_scale_matrix(center: torch.tensor, scale_factor: Union[float, torch.tensor]):
    if isinstance(scale_factor,
                  float) or scale_factor.dim() == 1:  # if only one value is provided it is scaled equally on both axes
        scale_factor = torch.ones(2).to(device) * scale_factor
    matrix = torch.zeros(2, 3).to(device)
    matrix = config_scale_matrix(scale_factor, center, matrix)
    return matrix


def get_squared_scale_matrix(center: torch.tensor, scale_factor: Union[float, torch.tensor]):
    if isinstance(scale_factor,
                  float) or scale_factor.dim() == 1:  # if only one value is provided it is scaled equally on both axes
        scale_factor = torch.ones(2).to(device) * scale_factor
    matrix = torch.zeros(3, 3).to(device)
    matrix = config_scale_matrix(scale_factor, center, matrix)
    return matrix


def scale_image(img: torch.tensor, scale_factor: torch.tensor, center: torch.tensor) -> torch.tensor:
    return kornia.geometry.scale(img, scale_factor=scale_factor, center=center)


def scale_coordinates_matrix(matrix_coordinates: torch.tensor, matrix: torch.tensor) -> torch.tensor:
    return torch.matmul(matrix, matrix_coordinates)


@prepare_data
def scale_compose_data(data: dict, scale_factor: Union[float, torch.tensor],
                       center: Union[torch.tensor, None] = None) -> dict:
    """

    :param data: dict of elements to be transformed
    :param scale_factor: factor of scaling
    :param center: center of scaling. By default its taken the center of the image
    :return: transformed data
    """
    ppal = 'data_2d'
    if len(data['data_2d']) == 0:
        ppal = 'data_2d_discrete'
    scale_factor = (torch.ones(1) * scale_factor).to(device)
    if center is None:
        center = utils.get_torch_image_center(data[ppal])
    center = center.to(device)
    matrix = get_scale_matrix(center, scale_factor)
    if len(data['data_2d']) != 0:
        data['data_2d'] = affine_image(data['data_2d'], matrix)
    if 'data_2d_discrete' in data:
        data['data_2d_discrete'] = affine_image(data['data_2d_discrete'], matrix, interpolation='nearest')
    if 'points_matrix' in data:
        data['points_matrix'] = scale_coordinates_matrix(data['points_matrix'], matrix)
    return data


""" --- Translation transform ---"""


def translate_image(img: torch.tensor, translation: torch.tensor) -> torch.tensor:
    return kornia.geometry.translate(img, translation)


def get_translation_matrix(translation: torch.tensor) -> torch.tensor:
    matrix = identity.clone()
    translation_x = translation[0] * one_torch
    translation_y = translation[1] * one_torch
    matrix[0, 2] = translation_x
    matrix[1, 2] = translation_y
    return matrix


def translate_coordinates_matrix(matrix_coordinates: torch.tensor, translation: torch.tensor) -> torch.tensor:
    matrix = torch.zeros((3, matrix_coordinates.shape[1])).to(device)
    row = torch.ones((1, matrix_coordinates.shape[1])).to(device)
    matrix[0] = row * translation[:, 0]
    matrix[1] = row * translation[:, 1]
    return matrix_coordinates + matrix


@prepare_data
def translate_compose_data(data: dict, translation: Union[int, torch.tensor]) -> dict:
    """

    :param data: dict of elements to be transformed
    :param translation: number of pixels to translate
    :return: transformed data
    """

    matrix = get_translation_matrix(translation)
    if not torch.is_tensor(translation):
        translation = (torch.tensor(translation).float().reshape((1, 2)))
    translation = translation.to(device)
    if len(data['data_2d']) != 0:
        data['data_2d'] = affine_image(data['data_2d'], matrix)
    if 'data_2d_discrete' in data:
        data['data_2d_discrete'] = affine_image(data['data_2d_discrete'], matrix, interpolation='nearest')
    if 'points_matrix' in data:
        data['points_matrix'] = translate_coordinates_matrix(data['points_matrix'], translation)
    return data


""" --- Shear Transform ---"""


def get_shear_matrix(shear_factor: tuple) -> torch.tensor:
    matrix = torch.eye(2, 3).to(device)
    matrix[0, 1] = shear_factor[0]
    matrix[1, 0] = shear_factor[1]
    return matrix


def get_squared_shear_matrix(shear_factor: tuple) -> torch.tensor:
    matrix = torch.eye(3, 3).to(device)
    matrix[0, 1] = shear_factor[0]
    matrix[1, 0] = shear_factor[1]
    return matrix


def shear_image(img: torch.tensor, shear_factor: torch.tensor) -> torch.tensor:
    return kornia.geometry.affine(img, shear_factor)


def shear_coordinates_matrix(matrix_coordinates: torch.tensor, matrix: torch.tensor) -> torch.tensor:
    return torch.matmul(matrix, matrix_coordinates)


@prepare_data
def shear_compose_data(data: dict, shear_factor: tuple) -> dict:
    """

    :param data: dict of elements to be transformed
    :param shear_factor: pixels of shearing
    :return: transformed data
    """
    matrix = get_shear_matrix(shear_factor)
    if len(data['data_2d']) != 0:
        data['data_2d'] = affine_image(data['data_2d'], matrix)
    if 'data_2d_discrete' in data:
        data['data_2d_discrete'] = affine_image(data['data_2d_discrete'], matrix, interpolation='nearest')
    if 'points_matrix' in data:
        data['points_matrix'] = shear_coordinates_matrix(data['points_matrix'], matrix)
    return data


def own_affine(tensor: torch.Tensor, matrix: torch.Tensor, interpolation: str = 'bilinear',
               padding_mode: str = 'border') -> torch.Tensor:
    """Apply an affine transformation to the image.

    :param tensor: The image tensor to be warped.
    :param matrix: The 2x3 affine transformation matrix.
    :param interpolation: interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
    :param padding_mode: padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.

    :return: The warped image.
    """
    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)
    matrix = matrix.expand(tensor.shape[0], -1, -1)
    # warp the input tensor
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = kornia.warp_affine(tensor, matrix, (height, width), flags=interpolation,
                                              padding_mode=padding_mode)
    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)
    return warped
