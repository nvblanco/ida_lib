import kornia
import numpy as np
import torch
import abc
from abc import ABCMeta

from image_augmentation import visualization
from . import utils

device = utils.device
data_types_2d = {"image", "mask", "heatmap"}
data_types_1d = {"keypoints"}

one_torch = torch.ones(1).to(device)


class transform(object):
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def __init__(self, data, visualize=False):
        self.visualize = visualize
        self.points = None
        result_data = {}
        if visualize:
            self.original = data
        if isinstance(data, dict):
            self.types_2d = {}
            compose_data = torch.tensor([])
            for type in data.keys():
                if type in data_types_2d:
                    #if type(data[type]).__module__ == np.__name_:
                    #    data[type] = kornia.image_to_tensor(data[type], keepdim=False)
                    compose_data = torch.cat((compose_data, data[type]),
                                             0)  # concatenate data into one multichannel pytoch tensor
                    self.types_2d[type] = data[type].shape[0]
                else:
                    self.points = data[type]
                    self.points_matrix = data[type]
            self.data2d = compose_data.to(device)
            # self.types2d= types_2d
            if self.points is not None: self.data1d = utils.keypoints_to_homogeneus_functional(self.points)
            if self.points is not None: self.points_matrix = utils.keypoints_to_homogeneus_and_concatenate(
                self.points_matrix)
            return result_data
        else:
            if data.dim() < 3:
                raise Exception("Single data must be al least 3 dims")
            else:
                self.data2d = data

    def postprocess_data(self):
        self.data2d = self.data2d.cpu()
        if self.types_2d is not None:
            data_output = {}
            data_split = torch.split(self.data2d, list(self.types_2d.values()), dim=0)
            for index, type in enumerate(self.types_2d):
                data_output[type] = data_split[index]
            if data_output.keys().__contains__('mask'): data_output['mask'] = utils.mask_change_to_01_functional(
                data_output['mask'])
            if self.__getattribute__('points_matrix') is not None: data_output['keypoints'] = [
                ((dato.cpu())[:2, :]).reshape(2) for dato in torch.split(self.points_matrix, 1, dim=1)]
        else:
            data_output = self.data2d
        if self.visualize:
            visualization.plot_image_tranformation(data_output, self.original)
        return data_output


class vflip_transformation(transform):
    def __init__(self, data, visualize=False):
        transform.__init__(self, data, visualize)

    def __call__(self, *args, **kwargs):
        self.data2d = kornia.vflip(self.data2d)
        if self.data1d is not None:
            heigth = self.data2d.shape[-2]
            if self.points_matrix is not None:
                self.points_matrix[1] = torch.ones(1, self.points_matrix.shape[1]).to(device) * (heigth) - \
                                        self.points_matrix[1]
        return transform.postprocess_data(self)


class hflip_transformation(transform):
    def __init__(self, data, visualize=False):
        transform.__init__(self, data, visualize)

    def __call__(self, *args, **kwargs):
        self.data2d = kornia.hflip(self.data2d)
        if self.points_matrix is not None:
            width = self.data2d.shape[-1]
            if self.points_matrix is not None:
                self.points_matrix[0] = torch.ones(1, self.points_matrix.shape[1]).to(device) * (width) - \
                                        self.points_matrix[0]
        return transform.postprocess_data(self)


class affine_transformation(transform):
    def __init__(self, data, matrix, visualize=False):
        transform.__init__(self, data, visualize)
        self.matrix = matrix.to(device)

    def __call__(self, *args, **kwargs):
        self.data2d = kornia.geometry.affine(self.data2d, self.matrix)
        if self.points_matrix is not None:
            self.points_matrix = torch.matmul(self.matrix, self.points_matrix)
        return transform.postprocess_data(self)


class rotate_transformation(transform):
    def __init__(self, data, degrees, visualize=False, center=None):
        transform.__init__(self, data, visualize)
        self.degrees = degrees * one_torch
        if center is None:
            self.center = torch.ones(1, 2)
            self.center[..., 0] = self.data2d.shape[-2] // 2  # x
            self.center[..., 1] = self.data2d.shape[-1] // 2  # y
        else:
            self.center = center
        self.center = self.center.to(device)

    def __call__(self, *args, **kwargs):
        self.data2d = kornia.geometry.rotate(self.data2d, angle=self.degrees, center=self.center)
        matrix = (
            kornia.geometry.get_rotation_matrix2d(angle=self.degrees, center=self.center, scale=one_torch)).reshape(2,
                                                                                                                    3)
        if self.points_matrix is not None:
            self.points_matrix = torch.matmul(matrix, self.points_matrix)
        return transform.postprocess_data(self)


class scale_transformation(transform):
    def __init__(self, data, scale_factor, visualize=False, center=None):
        transform.__init__(self, data, visualize)
        self.scale_factor = (torch.ones(1) * scale_factor).to(device)
        if center is None:
            self.center = torch.ones(1, 2)
            self.center[..., 0] = self.data2d.shape[-2] // 2  # x
            self.center[..., 1] = self.data2d.shape[-1] // 2  # y
        else:
            self.center = center
        self.center = self.center.to(device)

    def get_scale_matrix(self, center, scale_factor):
        if isinstance(scale_factor,
                      float) or scale_factor.dim() == 1:  # si solo se proporciona un valor; se escala por igual en ambos ejes
            scale_factor = torch.ones(2).to(device) * scale_factor
        matrix = torch.zeros(2, 3).to(device)
        matrix[0, 0] = scale_factor[0]
        matrix[1, 1] = scale_factor[1]
        matrix[0, 2] = (-scale_factor[0] + 1) * center[:, 0]
        matrix[1, 2] = (-scale_factor[1] + 1) * center[:, 1]
        return matrix

    def __call__(self, *args, **kwargs):
        self.data2d = kornia.geometry.scale(self.data2d, scale_factor=self.scale_factor, center=self.center)
        matrix = self.get_scale_matrix(self.center, self.scale_factor)
        if self.points_matrix is not None:
            self.points_matrix = torch.matmul(matrix, self.points_matrix)
        return transform.postprocess_data(self)


class translate_transformation(transform):
    def __init__(self, data, translation, visualize=False):
        transform.__init__(self, data, visualize)
        self.translation = translation
        if not torch.is_tensor(translation):
            self.translation = (torch.tensor(translation).float().reshape((1, 2)))
        self.translation = self.translation.to(device)

    def __aply_to_point(self, point):
        point[0] += self.translation[:, 0]
        point[1] += self.translation[:, 1]
        return point

    def __call__(self, *args, **kwargs):
        self.data2d = kornia.geometry.translate(self.data2d, self.translation)
        if self.points_matrix is not None:
            matrix = torch.zeros((3, self.points_matrix.shape[1])).to(device)
            row = torch.ones((1, self.points_matrix.shape[1])).to(device)
            matrix[0] = row * self.translation[:, 0]
            matrix[1] = row * self.translation[:, 1]
            self.points_matrix = self.points_matrix + matrix
        return transform.postprocess_data(self)

class shear_transformation(transform):
    def __init__(self, data, shear_factor, visualize = False):
        transform.__init__(self, data, visualize)
        self.shear_factor = shear_factor
