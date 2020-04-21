from abc import ABC, abstractmethod
import visualization
import torch
import torch.nn as nn
import kornia
import random
import functools
from operations import utils


device = 'cuda'
cuda = torch.device('cuda')
one_torch = torch.tensor(1,device=cuda)
one_torch = torch.ones(1, device=cuda)
ones_torch = torch.ones(1, 2, device=cuda)
data_types_2d = {"image", "mask", "heatmap"}
identity = torch.eye(3, 3, device=cuda)


def get_compose_matrix(operations):
    matrix =  identity.clone()
    for operation in operations:
        if operation.apply_according_to_probability():
            matrix = torch.matmul( operation.get_op_matrix(), matrix)
    return matrix


def split_operations_by_type(operations):
    color, geometry = [], []
    for op in operations:
        if op.get_op_type() == 'color': color.append(op)
        else: geometry.append(op)
    return color, geometry

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def get_compose_function(operations):
    funcs = [op.transform_function for op in operations if op.apply_according_to_probability]
    #compose_function = compose(tuple(funcs))
    compose_function = functools.reduce(lambda f, g: lambda x: f(g(x)), tuple(funcs), lambda x: x)
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(compose_function(i), 0, 255)
    return lookUpTable




class pipeline_operation(ABC):
    def __init__(self, type, probability = 1):
        self.probability = probability
        self.type = type

    @abstractmethod
    def get_op_matrix(self):
        pass

    def get_op_type(self):
        return self.type

    def apply_according_to_probability(self):
        return (random.randint(1,101) / 100 ) < self.probability


class contrast_pipeline(pipeline_operation):
    def __init__(self, probability, contrast_factor):
        pipeline_operation.__init__(self, probability=probability, type='color')
        self.contrast = contrast_factor

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x): return self.contrast * (x-127) +127

class brightness_pipeline(pipeline_operation):
    def __init__(self, probability, brightness_factor):
        pipeline_operation.__init__(self, probability=probability, type='color')
        self.brigthness = brightness_factor

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def get_op_type(self):
        return 'color'

    def transform_function(self, x): return x + self.brigthness

class gamma_pipeline(pipeline_operation):
    def __init__(self, probability, gamma_factor):
        pipeline_operation.__init__(self, probability=probability, type='color')
        self.gamma = gamma_factor

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x): return pow(x / 255.0, self.gamma) * 255.0

class normalize_pipeline(pipeline_operation):
    def __init__(self, probability, old_range = (0,255), new_range=(0,1)):
        pipeline_operation.__init__(self, probability=probability, type='color')
        self.new_range = new_range
        self.old_range = old_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x): return (x + self.old_range[0])/(self.old_range[1]-self.old_range[0])


class scale_pipeline(pipeline_operation):
    def __init__(self, probability, scale_factor, center=None, img_shape = (256,256,3)):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        if center is None:
            self.center = ones_torch.clone()
            self.center[..., 0] = img_shape[-2] // 2  # x
            self.center[..., 1] = img_shape[-1] // 2  # y
        else:
            self.center = center
        self.matrix = identity.clone()
        identity[0,0] = 55
        self.ones_2 = torch.ones(2, device=cuda)
        if isinstance(self.scale_factor,
                      float) or self.scale_factor.dim() == 1:  # si solo se proporciona un valor; se escala por igual en ambos ejes
            self.scale_factor = self.ones_2 * scale_factor
        else:
            self.scale_factor = scale_factor

    def get_op_matrix(self):
        self.matrix[0, 0] = self.scale_factor[0]
        self.matrix[1, 1] = self.scale_factor[1]
        self.matrix[0, 2] = (-self.scale_factor[0] + 1) * self.center[:, 0]
        self.matrix[1, 2] = (-self.scale_factor[1] + 1) * self.center[:, 1]
        return self.matrix


class rotate_pipeline(pipeline_operation):
    def __init__(self, probability, degrees, center=None, img_shape = (256,256,3)):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.degrees = degrees
        self.degrees = degrees * one_torch
        self.new_row =  torch.Tensor(1, 3).to(device)
        if center is None:
            self.center = ones_torch.clone()
            self.center[..., 0] = img_shape[-2] // 2  # x
            self.center[..., 1] = img_shape[-1] // 2  # y
        else:
            self.center = center

    def get_op_matrix(self):
        return torch.cat(((kornia.geometry.get_rotation_matrix2d(angle=self.degrees, center=self.center, scale=one_torch)).reshape(2,3), self.new_row))


class translate_pipeline(pipeline_operation):
    def __init__(self, probability, translation):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.translation_x = translation[0] * one_torch
        self.translation_y = translation[1] * one_torch
        self.matrix = identity.clone()

    def get_op_matrix(self):
        self.matrix[0,2]=self.translation_x
        self.matrix[1,2]=self.translation_y
        return self.matrix

class shear_pipeline(pipeline_operation):
    def __init__(self, probability, shear):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.shear_x = shear[0]
        self.shear_y = shear[1]
        self.matrix =  identity.clone()

    def get_op_matrix(self):
        self.matrix[0,1] = self.shear_x
        self.matrix[1,0] = self.shear_y
        return self.matrix

class hflip_pipeline(pipeline_operation):
    def __init__(self, probability, img_shape = (256,256,3)):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.width = img_shape[1]
        self.matrix =  identity.clone()
        self.matrix[0, 0] = -1
        self.matrix[0, 2] = self.width

    def get_op_matrix(self):
        return self.matrix

class vflip_pipeline(pipeline_operation):
    def __init__(self, probability, width = 256):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.matrix =  identity.clone()
        self.matrix[0, 0] = -1
        self.matrix[0, 2] = width

    def get_op_matrix(self):
        return self.matrix



def preprocess_dict_data_and_2dtypes(data):
    p_data = {}
    types_2d = {}
    compose_data = torch.tensor([])
    for type in data.keys():
        if type in data_types_2d:
            if data[type].dim()>3 : data[type] =  data[type][0,:]
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
    p_data['types_2d'] = {}
    compose_data = torch.tensor([])
    for type in data.keys():
        if type in data_types_2d:
            data[type] = kornia.image_to_tensor(data[type])
            if data[type].dim()>3 : data[type] =  data[type][0,:]
            compose_data = torch.cat((compose_data, data[type]),
                                     0)  # concatenate data into one multichannel pytoch tensor
            p_data['types_2d'][type] = data[type].shape[0]
        else:
            p_data['points_matrix'] = data[type]
    p_data['data_2d'] = compose_data.to(device)
    if p_data.keys().__contains__('points_matrix'):  p_data[
        'points_matrix'] = utils.keypoints_to_homogeneus_and_concatenate(
        p_data['points_matrix'])
    return p_data


def postprocess_data(batch):
    process_data = []
    for data in batch:
        if data.keys().__contains__('types_2d'):
            data_output = {}
            data_split = torch.split(data['data_2d'], list(data['types_2d'].values()), dim=0)
            for index, type in enumerate(data['types_2d']):
                data_output[type] = data_split[index]
            if data_output.keys().__contains__('mask'): data_output['mask'] = utils.mask_change_to_01_functional(
                data_output['mask'])
            if data.keys().__contains__('points_matrix'): data_output['keypoints'] = [
                ((dato)[:2, :]).reshape(2) for dato in torch.split(data['points_matrix'], 1, dim=1)]
        else:
            data_output = data['data_2d']
        process_data.append(data_output)
    return data_output

def postprocess_data_and_visualize(batch, data_original):
    process_data = []
    for data in batch:
        if data.keys().__contains__('types_2d'):
            data_output = {}
            data_split = torch.split(data['data_2d'], list(data['types_2d'].values()), dim=0)
            for index, type in enumerate(data['types_2d']):
                data_output[type] = data_split[index]
            if data_output.keys().__contains__('mask'): data_output['mask'] = utils.mask_change_to_01_functional(
                data_output['mask'])
            if data.keys().__contains__('points_matrix'): data_output['keypoints'] = [
                ((dato)[:2, :]).reshape(2) for dato in torch.split(data['points_matrix'], 1, dim=1)]
        else:
            data_output = data['data_2d']
        process_data.append(data_output)
    visualization.plot_batch_image_tranformation(process_data, data_original)
    return data_output

class pipeline(object):
    def __init__(self, pipeline_operations):
        self.color_ops, self.geom_ops = split_operations_by_type(pipeline_operations)

    def apply_geometry_transform_data2d(self, image, matrix):
        return kornia.geometry.affine(image, matrix[:2,:])

    def apply_geometry_transform_points(self, points_matrix, matrix):
        return torch.matmul(matrix, points_matrix)

    def __call__(self, batch_data, visualize = False):
        self.process_data = []
        for data in batch_data:
            lut = get_compose_function(self.color_ops)
            data['image'] = cv2.LUT(data['image'], lut)
            p_data = preprocess_dict_data(data)
            matrix = get_compose_matrix(self.geom_ops)
            p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
            p_data['points_matrix'] = self.apply_geometry_transform_points(p_data['points_matrix'], matrix)
            self.process_data.append(p_data)
        if visualize:
            return postprocess_data_and_visualize(self.process_data, batch_data)
        else:
            return postprocess_data(self.process_data)

import numpy as np
import cv2


img: np.ndarray = cv2.imread('../gato.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

keypoints = ([img.shape[0]//2, img.shape[1]//2], [img.shape[0]//2  + 15, img.shape[1]//2 - 50], [img.shape[0]//2  + 85, img.shape[1]//2 - 80], [img.shape[0]//2  - 105, img.shape[1]//2 +60])

points = [torch.from_numpy(np.asarray(point)) for point in keypoints]
#data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW




from operations import utils
#data = color.equalize_histogram(data, visualize=True)

data = {'image':img, 'keypoints': points}

batch = [data]

from time import time


start_time = time()
pip = pipeline(pipeline_operations=(
    contrast_pipeline(probability = 1, contrast_factor= 1.5),
    shear_pipeline(probability=0.9, shear = (0.01, 0.01)),
    translate_pipeline(probability=0.2, translation=(0.03,0.05)),
    brightness_pipeline(probability = 1, brightness_factor= 105),
    gamma_pipeline(probability = 1,gamma_factor= 3),
    shear_pipeline(probability=0.9, shear = (0.05, 0.01)),
    translate_pipeline(probability=0.2, translation=(0.03,0.05)),
    vflip_pipeline(probability=0.5),
    hflip_pipeline(probability=0.3)
))

batch = pip(batch, visualize=True)


operations = (brightness_pipeline(probability=1, brightness_factor=150),
              contrast_pipeline(probability=1, contrast_factor=2))

get_compose_function(operations)
consumed_time = time() - start_time
print(consumed_time)








