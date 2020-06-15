import functools
from typing import Optional, Union

import cv2
import kornia
import numpy as np
import torch

from ida_lib import visualization
from ida_lib.global_parameters import data_types_2d, device, internal_type, identity
from ida_lib.operations import utils
from ida_lib.operations.utils import remove_digits, add_new_axis

__all__ = ['get_compose_matrix',
           'get_compose_function',
           'preprocess_data',
           'split_operations_by_type',
           'postprocess_data',
           'switch_point_positions']

mask_types = []
other_types = []


def switch_point_positions(point_matrix, input_list):
    for exchange_list in input_list:  # not to do all the transformations together because there can be
        # circular transformations (1,5) (5,10)
        indices = [a for a, _ in exchange_list]
        indices2 = [b for _, b in exchange_list]
        exchange = point_matrix[..., indices].clone()
        point_matrix[..., indices] = point_matrix[..., indices2]
        point_matrix[..., indices2] = exchange
    return point_matrix


def get_compose_matrix(operations: list, data_info: Optional[dict] = None) -> torch.tensor:
    """
    Returns the transformation matrix composed by the multiplication in order of
    the input operations (according to their probability)
    If data_info is not None, go through the operations by entering the necessary information about the images
    (image center, shape..)

    :param operations: list of pipeline operations
    :param data_info: dict with data info to configure operations parameters
    :return: torch tensor of the transform matrix
    """
    matrix = identity.clone()
    switch = []
    if data_info:
        for operation in operations:
            if operation.need_data_info():
                operation.config_parameters(data_info)
            if operation.apply_according_to_probability():
                if operation.switch_points():
                    switch.append(operation.switch_points())
                matrix = torch.matmul(operation.get_op_matrix(), matrix)
    else:
        for operation in operations:
            if operation.apply_according_to_probability():
                if operation.switch_points():
                    switch.append(operation.switch_points())
                matrix = torch.matmul(operation.get_op_matrix(), matrix)
    return matrix, switch


def split_operations_by_type(operations: list) -> tuple:
    """
    Split input operations into sub-lists of each transformation type
    the normalization operation is placed last to apply correctly the other operations

    :param operations: list of pipeline operations
    :return: tuple of lists of the operations separated into color, geometry and independent
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
    if normalize is not None:
        color.append(normalize)  # normalization must be last color operation
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
    look_up_table = np.empty((1, 256), np.float)
    for i in range(256):
        look_up_table[0, i] = compose_function(i)
    look_up_table[0, :] = np.clip(look_up_table[0, :], 0, 255)
    return look_up_table


def preprocess_data(data: Union[list, dict], batch_info: Union[list, dict] = None, interpolation: str = None,
                    resize: Optional[tuple] = None) -> list:
    """
    Combines the 2d information in a tensor and the points in a homogeneous coordinate matrix \
    that allows applying the geometric operations in a single joint operation on the data \
    and another on the points.

    - Loads the data as tensor in GPU to prepare them as input to a neural network
    - Analyze the data info required for the transformations (shape, bpp...)
    - Resize the 2d data and keypoints to the new shape

    :param resize: if it is wanted to resize the data, indicate the new size
    :param data:  list of elements to be transformed through the pipe
    :param batch_info: dict with the required data info
    :param interpolation: desired interpolation mode to be applied
    :return: preprocessed and resized data, and dict with batch info
    """

    if batch_info is None:
        global mask_types
        global other_types
        mask_types = []
        other_types = []
        if resize is not None:
            return preprocess_dict_data_and_data_info_with_resize(data, resize[::-1], interpolation)
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

    :param data: list of elements to be transformed through the pipe
    :param new_size: desired output size for bidimensional data
    :param interpolation: desired interpolation mode to be applied
    :return: preprocessed and resized data, and dict with batch info
    """

    original_shape = None
    p_data = {}
    data_info = {'types_2d': {}, 'types_2d_discrete': {}, 'contains_keypoints': False}
    compose_data = torch.tensor([], dtype=internal_type)
    compose_discrete_data = torch.tensor([], dtype=internal_type)

    for actual_type in data:
        no_numbered_type = remove_digits(actual_type)
        if no_numbered_type in data_types_2d:
            if len(data[actual_type].shape) == 2:
                data[actual_type] = add_new_axis(data[actual_type])
            original_shape = data[actual_type].shape
            if actual_type not in data_types_2d:
                data_types_2d.add(actual_type)  # adds to the list of type names the numbered name detected
                # in the input data
            if 'shape' not in data_info:
                data_info['shape'] = (new_size[0], new_size[1], data[actual_type].shape[2])
                data_info['new_size'] = new_size
            data[actual_type] = (kornia.image_to_tensor(cv2.resize(data[actual_type], new_size))).type(
                internal_type)  # Transform to tensor + resize data
            if data[actual_type].dim() > 3:
                data[actual_type] = data[actual_type][0, :]
            if no_numbered_type == 'mask' or no_numbered_type == 'segmap':
                mask_types.append(actual_type)
                if interpolation != 'nearest':
                    compose_discrete_data = torch.cat((compose_discrete_data, data[actual_type]), 0)
                    data_info['types_2d_discrete'][actual_type] = data[actual_type].shape[0]
                else:
                    compose_data = torch.cat((compose_data, data[actual_type]),
                                             0)
                    data_info['types_2d'][actual_type] = data[actual_type].shape[0]
            else:
                data_info['types_2d'][actual_type] = data[actual_type].shape[0]
                compose_data = torch.cat((compose_data, data[actual_type]),
                                         0)  # concatenate data into one multichannel pytorch tensor
                data_info['types_2d'][actual_type] = data[actual_type].shape[0]
        elif no_numbered_type == 'keypoints':
            p_data['points_matrix'] = data[actual_type]
            data_info['contains_keypoints'] = True
        else:
            other_types.append(actual_type)
            p_data[actual_type] = data[actual_type]
    p_data['data_2d'] = compose_data.to(device)
    data_info['contains_discrete_data'] = len(mask_types) != 0
    if data_info['contains_discrete_data']:
        p_data['data_2d_discrete'] = compose_discrete_data.to(device)
    if 'points_matrix' in p_data:
        p_data['points_matrix'] = utils.keypoints_to_homogeneous_and_concatenate(
            p_data['points_matrix'], (new_size[0] / original_shape[1], new_size[1] / original_shape[0]))
    data_info['present_types'] = ([*data_info['types_2d']] + mask_types, other_types)
    return p_data, data_info


def preprocess_dict_data(data: list, batch_info: dict, resize: Optional[tuple] = None) -> list:
    """
    Combines the 2d information in a tensor and the points in a homogeneous coordinate matrix
    that allows applying the geometric operations in a single joint operation on the data
    and another on the points.
    * Loads the data as tensor in GPU to prepare them as input to a neural network

    :param data: list of elements to be transformed through the pipe
    :param batch_info: dict with necessary information about the batch data
    :param resize: whether to resize or not
    :return: preprocessed data
    """
    p_data = {}
    original_shape = None
    compose_data = torch.tensor([], dtype=internal_type)
    compose_discrete_data = torch.tensor([], dtype=internal_type)
    for actual_type in data:
        if actual_type in data_types_2d:
            if len(data[actual_type].shape) == 2:
                data[actual_type] = add_new_axis(data[actual_type])
            if resize:
                original_shape = data[actual_type].shape
                data[actual_type] = (kornia.image_to_tensor(
                    cv2.resize(data[actual_type], (batch_info['shape'][0], batch_info['shape'][1])))).type(
                    internal_type)
            else:
                data[actual_type] = (kornia.image_to_tensor(data[actual_type])).type(internal_type)
            if data[actual_type].dim() > 3:
                data[actual_type] = data[actual_type][0, :]
            if actual_type in mask_types and batch_info['contains_discrete_data']:
                compose_discrete_data = torch.cat((compose_discrete_data, data[actual_type]), 0)
            else:
                compose_data = torch.cat((compose_data, data[actual_type]),
                                         0)  # concatenate data into one multichannel pytorch tensor
        elif actual_type in other_types:
            p_data[actual_type] = data[actual_type]
        else:
            p_data['points_matrix'] = data[actual_type]
    p_data['data_2d'] = compose_data.to(device)
    if batch_info['contains_discrete_data']:
        p_data['data_2d_discrete'] = compose_discrete_data.to(device)
    resize_factor = None
    if resize:
        resize_factor = (batch_info['new_size'][0] / original_shape[1], batch_info['new_size'][1] / original_shape[0])
    if 'points_matrix' in p_data:
        p_data['points_matrix'] = utils.keypoints_to_homogeneous_and_concatenate(p_data['points_matrix'], resize_factor)
    return p_data


def preprocess_dict_data_and_data_info(data: list, interpolation: str) -> list:
    """
    It combines the 2d information in a tensor and the points in a homogeneous coordinate matrix
    that allows applying the geometric operations in a single joint operation on the data
    and another on the points.
        * Loads the data as tensor in GPU to prepare them as input to a neural network
        * Analyze the data info required for the transformations (shape, bpp...)
        * Add to the predetermined list of type names numbered names like 'mask2' to make possible to have multiple
            mask or elements of a single type

    :param data: list of elements to be transformed through the pipe
    :param interpolation: type of interpolation to be applied
    :return:preprocessed data and dict of data info
    """
    p_data = {}
    data_info = {'types_2d': {}, 'types_2d_discrete': {}, 'contains_keypoints': False}
    compose_data = torch.tensor([], dtype=internal_type)
    compose_discrete_data = torch.tensor([], dtype=internal_type)
    for actual_type in data:
        no_numbered_type = remove_digits(actual_type)
        if no_numbered_type in data_types_2d:
            if actual_type not in data_types_2d:
                data_types_2d.add(actual_type)  # adds to the list of type names the numbered
                # name detected in the input data
            if len(data[actual_type].shape) == 2:
                data[actual_type] = add_new_axis(data[actual_type])
            if 'shape' not in data_info:
                data_info['shape'] = data[actual_type].shape
            data[actual_type] = (kornia.image_to_tensor(data[actual_type])).type(internal_type)
            if data[actual_type].dim() > 3:
                data[actual_type] = data[actual_type][0, :]
            if no_numbered_type == 'mask' or no_numbered_type == 'segmap':
                mask_types.append(actual_type)
                if interpolation != 'nearest':
                    compose_discrete_data = torch.cat((compose_discrete_data, data[actual_type]), 0)
                    data_info['types_2d_discrete'][actual_type] = data[actual_type].shape[0]
                else:
                    compose_data = torch.cat((compose_data, data[actual_type]),
                                             0)
                    data_info['types_2d'][actual_type] = data[actual_type].shape[0]
            else:
                data_info['types_2d'][actual_type] = data[actual_type].shape[0]
                compose_data = torch.cat((compose_data, data[actual_type]),
                                         0)  # concatenate data into one multichannel pytorch tensor
                data_info['types_2d'][actual_type] = data[actual_type].shape[0]
        elif no_numbered_type == 'keypoints':
            p_data['points_matrix'] = data[actual_type]
            data_info['contains_keypoints'] = True
        else:
            other_types.append(actual_type)
            p_data[actual_type] = data[actual_type]
    p_data['data_2d'] = compose_data.to(device)
    data_info['contains_discrete_data'] = (len(mask_types) != 0 and interpolation != 'nearest')
    if data_info['contains_discrete_data']:
        p_data['data_2d_discrete'] = compose_discrete_data.to(device)
    if 'points_matrix' in p_data:
        p_data['points_matrix'] = utils.keypoints_to_homogeneous_and_concatenate(p_data['points_matrix'])
    data_info['present_types'] = ([*data_info['types_2d']] + mask_types, other_types)
    return p_data, data_info


def postprocess_data(batch: list, batch_info: dict, data_original: Optional[list] = None, visualize: bool = False,
                     original_type: torch.dtype = torch.uint8, output_format: str = 'dict') -> list:
    """
    Restores the data to the original form;
    separating the matrix into the different 2d input data and point coordinates.

    :param batch: list of elements to be transformed through the pipe
    :param batch_info: dict with necessary information about the batch data
    :param data_original: original batch before transforms
    :param visualize: whether to run the visualization tool or not
    :param original_type: torch original type of the input data to do the conversion of the output data to this type
    :param output_format: whether to format the output element as a dict or as a tuple
    :return: processed data
    """
    process_data = []
    discrete_data_split = None
    for data in batch:
        if 'types_2d' in batch_info:
            data_output = {}
            data_split = torch.split(data['data_2d'], list(batch_info['types_2d'].values()), dim=0)
            if batch_info['contains_discrete_data']:
                discrete_data_split = torch.split(data['data_2d_discrete'], list(batch_info['types_2d_discrete']
                                                                                 .values()), dim=0)
            for index, actual_type in enumerate(batch_info['types_2d']):
                data_output[actual_type] = data_split[index].type(original_type)
            for index, actual_type in enumerate(batch_info['types_2d_discrete']):
                data_output[actual_type] = discrete_data_split[index].type(original_type)
            for label in other_types:
                data_output[label] = data[label]
            if 'points_matrix' in data:
                data_output['keypoints'] = utils.homogeneous_points_to_matrix(data['points_matrix'])
        else:
            data_output = data['data_2d']
        process_data.append(data_output)
    if visualize:
        visualization.visualize(process_data[0:5], data_original[0:5], mask_types)
    if len(process_data) == 1:  # if the output is a single item, the list is removed
        process_data = process_data[0]
        if output_format == 'tuple':
            return tuple(process_data.values())
        return process_data
    else:
        if output_format == 'tuple':
            return [tuple(d.values()) for d in process_data]
        return process_data
