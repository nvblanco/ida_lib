from typing import Optional

import cv2
import functools
from string import digits
import numpy as np
from ida_lib.operations import utils
import torch
import kornia
from kornia.geometry.transform.imgwarp import warp_affine
from ida_lib.core import visualization


__all__ = ['get_compose_matrix',
           'get_compose_function',
           'preprocess_data',
           'preprocess_dict_data',
           'preprocess_dict_data_and_data_info',
           'preprocess_dict_data_and_data_info_with_resize',
           'split_operations_by_type',
           'get_compose_matrix_and_configure_parameters',
           'postprocess_data',
           'own_affine']


device = 'cuda'
cuda = torch.device('cuda')
data_types_2d = {"image", "mask", "segmap", "heatmap"}
present_2d_types = []
mask_types = []
other_types = []
internal_type = torch.float32
identity = torch.eye(3, 3, device=cuda)

def get_compose_matrix(operations: list) -> torch.tensor:
    """
    Returns the transformation matrix composed by the multiplication in order of
    the input operations (according to their probability)

    :param operations : list of pipeline operations
    :return : torch tensor of the transform matrix
    """
    matrix = identity.clone()
    for operation in operations:
        if operation.apply_according_to_probability():
            matrix = torch.matmul(operation.get_op_matrix(), matrix)
    return matrix



def get_compose_matrix_and_configure_parameters(operations: list, data_info: dict) -> torch.tensor:
    """
    Returns the transformation matrix composed by the multiplication in order of
    the input operations (according to their probability).

    Go through the operations by entering the necessary information about the images (image center, shape..)
    :param operations: list of pipeline operations
    :param data_info : dict with data info to configure operations parameters
    :return : torch tensor of the transform matrix
    """
    matrix = identity.clone()
    for operation in operations:
        if operation.need_data_info():
            operation.config_parameters(data_info)
        if operation.apply_according_to_probability():
            matrix = torch.matmul(operation.get_op_matrix(), matrix)
    return matrix


def own_affine(tensor: torch.Tensor, matrix: torch.Tensor, interpolation: str = 'bilinear',
               padding_mode: str = 'border') -> torch.Tensor:
    """Apply an affine transformation to the image.

    :param tensor :     The image tensor to be warped.
    :param matrix :     The 2x3 affine transformation matrix.
    :param interpolation : interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
    :param padding_mode : padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.

    :return : The warped image.
    """
    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)
    matrix = matrix.expand(tensor.shape[0], -1, -1)
    # warp the input tensor
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = warp_affine(tensor, matrix, (height, width), flags=interpolation, padding_mode=padding_mode)
    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)
    return warped



def split_operations_by_type(operations: list) -> tuple:
    """
    Split input operations into sub-lists of each transformation type
*   the normalization operation is placed last to apply correctly the other operations
    :param operations: list of pipeline operations
    :return : tuple of lists of the operations separated into color, geometry and independent
    """
    color, geometry, independent = [], [], []
    normalize = None
    for op in operations:
        if op.get_op_type() == 'color':
            color.append(op)
        elif op.get_op_type() == 'geometry':
            geometry.append(op)
        elif op.get_op_type() == 'normalize':
            normalize = op
        else:
            independent.append(op)
    if normalize is not None: color.append(normalize)  # normalization must be last color operation
    return color, geometry, independent


def compose(*functions):
    """
    Return lambda function that represent the composition of the input functions
    :param functions: math functions to be composed
    :return: compose function
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def get_compose_function(operations: list) -> np.ndarray:
    """
    returns the LUT table with the correspondence of each possible value
    according to the color operations to be implemented (according to their probability)
    :param operations: list of pipeline operations
    :return: compose function
    """
    funcs = [op.transform_function for op in operations if op.apply_according_to_probability()]
    compose_function = functools.reduce(lambda f, g: lambda x: f(g(x)), tuple(funcs), lambda x: x)
    lookUpTable = np.empty((1, 256), np.int16)
    for i in range(256):
        lookUpTable[0, i] = compose_function(i)
    lookUpTable[0, :] = np.clip(lookUpTable[0, :], 0, 255)
    return np.uint8(lookUpTable)


def preprocess_data(data:list, batch_info: dict=None,  new_size: tuple=None, interpolation: str=None, resize:bool=False) ->list:
    """
    Combines the 2d information in a tensor and the points in a homogeneous coordinate matrix
    that allows applying the geometric operations in a single joint operation on the data
    and another on the points.
     * Loads the data as tensor in GPU to prepare them as input to a neural network
     * Analyze the data info required for the transformations (shape, bpp...)
     * Resize the 2d data and keypoints to the new shape
    :param data:        list of elements to be tranformed through the pipe
    :param new_size :   desired output size for bidimensional data
    :param interpolation: desired interpolation mode to be applied
    :return: preprocessed and resized data, and dict with batch info
    """
    if batch_info is None:
        if resize is not None:
            return preprocess_dict_data_and_data_info_with_resize(data, new_size, interpolation)
        else:
            return preprocess_dict_data_and_data_info(data, interpolation)
    else:
        return preprocess_dict_data(data, batch_info, resize)



def preprocess_dict_data_and_data_info_with_resize(data: list, new_size: tuple, interpolation: str) -> list:
    """
    Combines the 2d information in a tensor and the points in a homogeneous coordinate matrix
    that allows applying the geometric operations in a single joint operation on the data
    and another on the points.
     * Loads the data as tensor in GPU to prepare them as input to a neural network
     * Analyze the data info required for the transformations (shape, bpp...)
     * Resize the 2d data and keypoints to the new shape
    :param data:        list of elements to be tranformed through the pipe
    :param new_size :   desired output size for bidimensional data
    :param interpolation: desired interpolation mode to be applied
    :return: preprocessed and resized data, and dict with batch info
    """
    p_data = {}
    data_info = {}
    data_info['types_2d'] = {}
    data_info['types_2d_discreted'] = {}
    data_info['contains_keypoints'] = False
    compose_data = torch.tensor([], dtype=internal_type)
    compose_discretized_data = torch.tensor([], dtype=internal_type)
    remove_digits = str.maketrans('', '', digits)
    for actual_type in data.keys():
        no_numbered_type = actual_type.translate(remove_digits)
        if no_numbered_type in data_types_2d:
            original_shape = data[actual_type].shape
            if not actual_type in data_types_2d: data_types_2d.add(
                actual_type)  # adds to the list of type names the numbered name detected in the input data
            if not 'shape' in data_info:
                data_info['shape'] = (new_size[0], new_size[1], data[actual_type].shape[2])
                data_info['bpp'] = data[actual_type].dtype
                """bpp = (data[type].dtype)[4:]"""
                data_info['new_size'] = new_size
            data[actual_type] = (kornia.image_to_tensor(cv2.resize(data[actual_type], new_size))).type(internal_type)  # Transform to tensor + resize data
            if data[actual_type].dim() > 3: data[actual_type] = data[actual_type][0, :]
            if (no_numbered_type == 'mask' or no_numbered_type == 'segmap'):
                mask_types.append(actual_type)
                if interpolation != 'nearest':
                    compose_discretized_data = torch.cat((compose_discretized_data, data[actual_type]),
                                                         0)
                    data_info['types_2d_discreted'][actual_type] = data[actual_type].shape[0]
                else:
                    compose_data = torch.cat((compose_data, data[actual_type]),
                                             0)
                    data_info['types_2d'][actual_type] = data[actual_type].shape[0]
            else:
                data_info['types_2d'][actual_type] = data[actual_type].shape[0]
                compose_data = torch.cat((compose_data, data[actual_type]),
                                         0)  # concatenate data into one multichannel pytoch tensor
                data_info['types_2d'][actual_type] = data[actual_type].shape[0]
        elif no_numbered_type == 'keypoints':
            p_data['points_matrix'] = data[actual_type]
            data_info['contains_keypoints'] = True
        else:
            other_types.append(actual_type)
            p_data[actual_type] = data[actual_type]
    p_data['data_2d'] = compose_data.to(device)
    data_info['contains_discrete_data'] = mask_types.__len__() != 0
    if data_info['contains_discrete_data']:
        p_data['data_2d_discreted'] = compose_discretized_data.to(device)
    if 'points_matrix' in p_data:
        p_data['points_matrix'] = utils.keypoints_to_homogeneus_and_concatenate(
        p_data['points_matrix'], (new_size[0] / original_shape[1], new_size[1] / original_shape[0]))
    return p_data, data_info


def preprocess_dict_data(data: list, batch_info: dict, resize:bool = False) -> list:
    """
    Combines the 2d information in a tensor and the points in a homogeneous coordinate matrix
    that allows applying the geometric operations in a single joint operation on the data
    and another on the points.
    * Loads the data as tensor in GPU to prepare them as input to a neural network
    :param data             (list)  : list of elements to be tranformed through the pipe
    :param batch_info       (dict)  : dict with necesary information about the batch data
    :return: preprocessed data
    """
    p_data = {}
    compose_data = torch.tensor([])
    compose_discretized_data = torch.tensor([])
    for actual_type in data.keys():
        if actual_type in data_types_2d:
            if resize:
                original_shape = data[actual_type].shape
                data[actual_type] = (kornia.image_to_tensor(
                    cv2.resize(data[actual_type], (batch_info['shape'][0], batch_info['shape'][1])))).type(
                    internal_type)
            else:
                data[actual_type] = (kornia.image_to_tensor(data[actual_type])).type(internal_type)
            if data[actual_type].dim() > 3: data[actual_type] = data[actual_type][0, :]
            if actual_type in mask_types and batch_info['contains_discrete_data']:
                compose_discretized_data = torch.cat((compose_discretized_data, data[actual_type]),
                                                     0)
            else:
                compose_data = torch.cat((compose_data, data[actual_type]),
                                         0)  # concatenate data into one multichannel pytoch tensor
        elif actual_type in other_types:
            p_data[actual_type] = data[actual_type]
        else:
            p_data['points_matrix'] = data[actual_type]
    p_data['data_2d'] = compose_data.to(device)
    if batch_info['contains_discrete_data']:
        p_data['data_2d_discreted'] = compose_discretized_data.to(device)
    resize_factor = None
    if resize:
        resize_factor = (batch_info['new_size'][0] / original_shape[1], batch_info['new_size'][1] / original_shape[0])
        '''if 'points_matrix' in p_data:  p_data[
            'points_matrix'] = utils.keypoints_to_homogeneus_and_concatenate_with_resize(
            p_data['points_matrix'],
            (batch_info['new_size'][0] / original_shape[1], batch_info['new_size'][1] / original_shape[0]))
    else:'''
    if 'points_matrix' in p_data:  p_data['points_matrix'] = utils.keypoints_to_homogeneus_and_concatenate( p_data['points_matrix'], resize_factor)
    return p_data


def preprocess_dict_data_and_data_info(data: list, interpolation: str) -> list:
    """
    It combines the 2d information in a tensor and the points in a homogeneous coordinate matrix
    that allows applying the geometric operations in a single joint operation on the data
    and another on the points.
        * Loads the data as tensor in GPU to prepare them as input to a neural network
        * Analyze the data info required for the transformations (shape, bpp...)
        * Add to the predetermined list of type names numbered names like 'mask2' to make posible to have multiple mask or elements of a single type
    :param data: list of elements to be tranformed through the pipe
    :param interpolation: type of interpolation to be applied
    :return: preprocesed data and dict of data info
    """
    p_data = {}
    data_info = {}
    data_info['types_2d'] = {}
    data_info['types_2d_discreted'] = {}
    data_info['contains_keypoints'] = False
    compose_data = torch.tensor([], dtype=internal_type)
    compose_discretized_data = torch.tensor([], dtype = internal_type)
    remove_digits = str.maketrans('', '', digits)
    for actual_type in data.keys():
        no_numbered_type = actual_type.translate(remove_digits)
        if no_numbered_type in data_types_2d:
            if not actual_type in data_types_2d: data_types_2d.add(
                actual_type)  # adds to the list of type names the numbered name detected in the input data

            if not data_info.keys().__contains__('shape'):
                data_info['shape'] = data[actual_type].shape
                data_info['bpp'] = data[actual_type].dtype
            data[actual_type] = (kornia.image_to_tensor(data[actual_type])).type(internal_type)
            if data[actual_type].dim() > 3: data[actual_type] = data[actual_type][0, :]
            if (no_numbered_type == 'mask' or no_numbered_type == 'segmap'):
                mask_types.append(actual_type)
                if interpolation != 'nearest':
                    compose_discretized_data = torch.cat((compose_discretized_data, data[actual_type]),
                                                         0)
                    data_info['types_2d_discreted'][actual_type] = data[actual_type].shape[0]
                else:
                    compose_data = torch.cat((compose_data, data[actual_type]),
                                             0)
                    data_info['types_2d'][actual_type] = data[actual_type].shape[0]
            else:
                data_info['types_2d'][actual_type] = data[actual_type].shape[0]
                compose_data = torch.cat((compose_data, data[actual_type]),
                                         0)  # concatenate data into one multichannel pytoch tensor
                data_info['types_2d'][actual_type] = data[actual_type].shape[0]
        elif no_numbered_type == 'keypoints':
            p_data['points_matrix'] = data[actual_type]
            data_info['contains_keypoints'] = True
        else:
            other_types.append(actual_type)
            p_data[actual_type] = data[actual_type]
    p_data['data_2d'] = compose_data.to(device)
    data_info['contains_discrete_data'] = (mask_types.__len__() != 0 and interpolation != 'nearest')
    if data_info['contains_discrete_data']:
        p_data['data_2d_discreted'] = compose_discretized_data.to(device)
    if 'points_matrix' in p_data:  p_data[
        'points_matrix'] = utils.keypoints_to_homogeneus_and_concatenate(
        p_data['points_matrix'])
    data_info['present_types'] = ([*data_info['types_2d']] + mask_types , other_types)
    return p_data, data_info



def postprocess_data(batch: list, batch_info: dict,  data_original: Optional[list]=None, visualize:bool = False) -> list:
    """
    Restores the data to the original form; separating the matrix into the different 2d input data and point coordinates.
    :param batch        (list) : list of elements to be tranformed through the pipe
    :param batch_info   (dict) : dict with necesary information about the batch data
    :return: processed data
    """
    process_data = []
    for data in batch:
        if batch_info.keys().__contains__('types_2d'):
            data_output = {}
            data_split = torch.split(data['data_2d'], list(batch_info['types_2d'].values()), dim=0)
            if batch_info['contains_discrete_data']: discreted_data_split = torch.split(data['data_2d_discreted'], list(
                batch_info['types_2d_discreted'].values()), dim=0)
            for index, actual_type in enumerate(batch_info['types_2d']):
                data_output[actual_type] = data_split[index]
            for index, actual_type in enumerate(batch_info['types_2d_discreted']):
                data_output[actual_type] = discreted_data_split[index]
            """for mask in mask_types: 
                data_output[mask] = utils.mask_change_to_01_functional(
                data_output[mask])"""
            for label in other_types: data_output[label] = data[label]
            """if data.keys().__contains__('points_matrix'): data_output['keypoints'] = [
                ((dato)[:2, :]).reshape(2) for dato in torch.split(data['points_matrix'], 1, dim=1)]"""
            if 'points_matrix' in data: data_output['keypoints'] = utils.homogeneus_points_to_matrix(data['points_matrix'])
        else:
            data_output = data['data_2d']
        process_data.append(data_output)
    if visualize:
        visualization.visualize(process_data[0:10], data_original[0:10], mask_types, other_types)
    return process_data

