
import visualization
import torch
import functools

from core.pipeline_operations import *

device = 'cuda'
cuda = torch.device('cuda')
one_torch = torch.tensor(1, device=cuda)
one_torch = torch.ones(1, device=cuda)
ones_torch = torch.ones(1, 2, device=cuda)
data_types_2d = {"image", "mask", "heatmap"}
identity = torch.eye(3, 3, device=cuda)

def get_compose_matrix(operations):
    matrix = identity.clone()
    for operation in operations:
        if operation.apply_according_to_probability():
            matrix = torch.matmul(operation.get_op_matrix(), matrix)
    return matrix


def get_compose_matrix_and_configure_parameters(operations, data_info):
    matrix = identity.clone()
    for operation in operations:
        if operation.need_data_info():
            operation.config_parameters(data_info)
        if operation.apply_according_to_probability():
            matrix = torch.matmul(operation.get_op_matrix(), matrix)
    return matrix


def split_operations_by_type(operations):
    color, geometry, independent = [], [],  []
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
    if normalize is not None: color.append(normalize) #normalization must be last color operation
    return color, geometry, independent


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def get_compose_function(operations):
    funcs = [op.transform_function for op in operations if op.apply_according_to_probability()]
    # compose_function = compose(tuple(funcs))
    compose_function = functools.reduce(lambda f, g: lambda x: f(g(x)), tuple(funcs), lambda x: x)
    lookUpTable = np.empty((1, 256), np.int16)
    for i in range(256):
        lookUpTable[0, i] = compose_function(i)
    lookUpTable[0, :] = np.clip(lookUpTable[0, :], 0, 255)
    return np.uint8(lookUpTable)


def preprocess_dict_data_and_2dtypes(data):
    p_data = {}
    types_2d = {}
    compose_data = torch.tensor([])
    for type in data.keys():
        if type in data_types_2d:
            if data[type].dim() > 3: data[type] = data[type][0, :]
            compose_data = torch.cat((compose_data, data[type]),
                                     0)  # concatenate data into one multichannel pytoch tensor
            types_2d[type] = data[type].shape[0]
        else:
            p_data['points_matrix'] = data[type]
    p_data['data_2d'] = compose_data.to(device)
    if p_data.keys().__contains__('points_matrix'):  p_data[
        'points_matrix'] = utils.keypoints_to_homogeneus_and_concatenate(
        p_data['points_matrix'])
    return p_data, types_2d


def preprocess_dict_data(data):
    p_data = {}
    compose_data = torch.tensor([])
    for type in data.keys():
        if type in data_types_2d:
            data[type] = kornia.image_to_tensor(data[type])
            if data[type].dim() > 3: data[type] = data[type][0, :]
            compose_data = torch.cat((compose_data, data[type]),
                                     0)  # concatenate data into one multichannel pytoch tensor
        else:
            p_data['points_matrix'] = data[type]
    p_data['data_2d'] = compose_data.to(device)
    if p_data.keys().__contains__('points_matrix'):  p_data[
        'points_matrix'] = utils.keypoints_to_homogeneus_and_concatenate(
        p_data['points_matrix'])
    return p_data


def preprocess_dict_data_and_data_info(data):
    p_data = {}
    data_info = {}
    data_info['types_2d'] = {}
    compose_data = torch.tensor([])
    for type in data.keys():
        if type in data_types_2d:
            if not data_info.keys().__contains__('shape'):
                data_info['shape'] = data[type].shape
                data_info['bpp'] = data[type].dtype
            data[type] = kornia.image_to_tensor(data[type])
            if data[type].dim() > 3: data[type] = data[type][0, :]
            compose_data = torch.cat((compose_data, data[type]),
                                     0)  # concatenate data into one multichannel pytoch tensor
            data_info['types_2d'][type] = data[type].shape[0]
        else:
            p_data['points_matrix'] = data[type]
    p_data['data_2d'] = compose_data.to(device)
    if p_data.keys().__contains__('points_matrix'):  p_data[
        'points_matrix'] = utils.keypoints_to_homogeneus_and_concatenate(
        p_data['points_matrix'])
    return p_data, data_info


def postprocess_data(batch, batch_info):
    process_data = []
    for data in batch:
        if batch_info.keys().__contains__('types_2d'):
            data_output = {}
            data_split = torch.split(data['data_2d'], list(batch_info['types_2d'].values()), dim=0)
            for index, type in enumerate(batch_info['types_2d']):
                data_output[type] = data_split[index]
            if data_output.keys().__contains__('mask'): data_output['mask'] = utils.mask_change_to_01_functional(
                data_output['mask'])
            if data.keys().__contains__('points_matrix'): data_output['keypoints'] = [
                ((dato)[:2, :]).reshape(2) for dato in torch.split(data['points_matrix'], 1, dim=1)]
        else:
            data_output = data['data_2d']
        process_data.append(data_output)
    return process_data


def postprocess_data_and_visualize(batch, data_original, batch_info):
    process_data = []
    for data in batch:
        if batch_info.keys().__contains__('types_2d'):
            data_output = {}
            data_split = torch.split(data['data_2d'], list(batch_info['types_2d'].values()), dim=0)
            for index, type in enumerate(batch_info['types_2d']):
                data_output[type] = data_split[index]
            if data_output.keys().__contains__('mask'): data_output['mask'] = utils.mask_change_to_01_functional(
                data_output['mask'])
            if data.keys().__contains__('points_matrix'): data_output['keypoints'] = [
                ((dato)[:2, :]).reshape(2) for dato in torch.split(data['points_matrix'], 1, dim=1)]
        else:
            data_output = data['data_2d']
        process_data.append(data_output)
    visualization.visualize(process_data[0:10], data_original[0:10])
    return process_data


class pipeline(object):
    def __init__(self, pipeline_operations):
        self.color_ops, self.geom_ops, self.indep_ops = split_operations_by_type(pipeline_operations)
        self.geom_ops.reverse()  # to apply matrix multiplication in the user order
        self.info_data = None

    def apply_geometry_transform_data2d(self, image, matrix):
        return kornia.geometry.affine(image, matrix[:2, :])

    def apply_geometry_transform_points(self, points_matrix, matrix):
        return torch.matmul(matrix, points_matrix)

    def __call__(self, batch_data, visualize=False):
        if visualize: original = [d.copy() for d in batch_data]
        self.process_data = []
        if self.info_data is None:  # First iteration to configure parameters and scan data info while the first item is being processed
            data = batch_data[0]
            batch_data = batch_data[1:]  # exclude the first item in the batch to be processed on the second loop
            lut = get_compose_function(self.color_ops)
            data['image'] = cv2.LUT(data['image'], lut)
            for op in self.indep_ops: data['image'] = op._apply_to_image_if_probability(data['image'])
            p_data, self.info_data = preprocess_dict_data_and_data_info(data)
            matrix = get_compose_matrix_and_configure_parameters(self.geom_ops, self.info_data)
            p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
            p_data['points_matrix'] = self.apply_geometry_transform_points(p_data['points_matrix'], matrix)
            self.process_data.append(p_data)

        for data in batch_data:
            lut = get_compose_function(self.color_ops)
            data['image'] = cv2.LUT(data['image'], lut)
            for op in self.indep_ops: data['image'] = op._apply_to_image_if_probability(data['image'])
            p_data = preprocess_dict_data(data)
            matrix = get_compose_matrix(self.geom_ops)
            p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
            p_data['points_matrix'] = self.apply_geometry_transform_points(p_data['points_matrix'], matrix)
            self.process_data.append(p_data)
        if visualize:
            return postprocess_data_and_visualize(self.process_data, original, self.info_data)
        else:
            return postprocess_data(self.process_data, self.info_data)


import numpy as np
import cv2

img: np.ndarray = cv2.imread('../bird.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

keypoints = ([img.shape[0] // 2, img.shape[1] // 2], [img.shape[0] // 2 + 15, img.shape[1] // 2 - 50],
             [img.shape[0] // 2 + 85, img.shape[1] // 2 - 80], [img.shape[0] // 2 - 105, img.shape[1] // 2 + 60])

points = [torch.from_numpy(np.asarray(point)) for point in keypoints]
# data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW


# data = color.equalize_histogram(data, visualize=True)

data = {'image': img, 'keypoints': points}
samples = 50

batch = [data.copy() for _ in range(samples)]
batch2 = [data.copy() for _ in range(samples)]

from time import time



start_time = time()
pip = pipeline(pipeline_operations=(
    translate_pipeline(probability=0, translation=(3, 0.05)),
    vflip_pipeline(probability=0),
    hflip_pipeline(probability=0),
    contrast_pipeline(probability=0, contrast_factor=1),
    gaussian_noise_pipeline(probability = 0, var=10),
    salt_and_pepper_noise_pipeline(probability=0, amount=1),
    spekle_noise_pipeline(probability=0),
    poisson_noise_pipeline(probability=0),
    random_brightness_pipeline(probability=0, brightness_range=(1.5, 1.6)),
    gamma_pipeline(probability=0, gamma_factor=0.5),
    random_translate_pipeline(probability=1, translation_range=(-90,90)),
    # normalize_pipeline(),
    # desnormalize_pipeline(),
    #resize_pipeline(probability=1, new_size=(50, 50)),
    random_scale_pipeline(probability=0, scale_range=(0.5, 1.5), center_desviation=20),
    random_rotate_pipeline(probability=0, degrees_range=(-50, 50), center_desviation=20),
    random_translate_pipeline(probability=0, translation_range=(20, 100)),
    random_shear_pipeline(probability=0, shear_range=(0, 0.5))
))
''',
  shear_pipeline(probability=0.9, shear = (0.01, 0.01)),


  shear_pipeline(probability=0.9, shear = (0.05, 0.01)),
  translate_pipeline(probability=0.2, translation=(0.03,0.05)),
  vflip_pipeline(probability=1),
  hflip_pipeline(probability=1),
  gaussian_noisse_pipeline(probability=0),
  salt_and_pepper_noisse_pipeline(probability=0, amount = 0.1),
  spekle_noisse_pipeline(probability=0),
  poisson_noisse_pipeline(probability=0),
  gaussian_blur_pipeline(probability=0),
  blur_pipeline(probability=1)'''
batch = pip(batch, visualize=True)

operations = (brightness_pipeline(probability=1, brightness_factor=150),
              contrast_pipeline(probability=1, contrast_factor=2))

get_compose_function(operations)
consumed_time = time() - start_time
print(consumed_time)
print(consumed_time / samples)
