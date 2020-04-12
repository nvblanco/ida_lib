import kornia
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
                    compose_data = torch.cat((compose_data, data[type]),
                                             0)  # concatenate data into one multichannel pytoch tensor
                    self.types_2d[type] = data[type].shape[0]
                else:
                    self.points = data[type]
            self.data2d = compose_data.to(device)
            #self.types2d= types_2d
            if self.points is not None: self.data1d = utils.keypoints_to_homogeneus_functional(self.points)
            return result_data
        else:
            if data.dim() < 3:
                raise Exception("Single data must be al least 3 dims")
            else:
                self.data2d = data

    def postprocess_data(self):
        self.data2d = self.data2d .cpu()
        if self.types_2d is not None:
            data_output = {}
            data_split = torch.split(self.data2d,  list(self.types_2d.values()), dim=0)
            for index, type in enumerate(self.types_2d):
                data_output[type] = data_split[index]
            data_output['keypoints'] = [((dato.cpu())[:2, :]).reshape(2) for dato in self.data1d]
            #if data_output.keys().__contains__('mask'): data_output['mask'] = mask_change_to_01_functional(data_output['mask'])
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
            for point in self.data1d:
                point[1] = heigth - point[1]
        return transform.postprocess_data(self)

class hflip_transformation(transform):
    def __init__(self, data, visualize=False):
        transform.__init__(self, data, visualize)

    def __call__(self, *args, **kwargs):
        self.data2d = kornia.hflip(self.data2d)
        if self.data1d is not None:
            width = self.data2d.shape[-1]
            for point in self.data1d:
                point[0] = width - point[0]
        return transform.postprocess_data(self)

class affine_transformation(transform):
    def __init__(self, data,matrix, visualize=False):
        transform.__init__(self, data, visualize)
        self.matrix = matrix.to(device)

    def __call__(self, *args, **kwargs):
        kornia.geometry.affine(self.data2d, self.matrix)
        if self.data1d is not None:
            self.data1d = [torch.matmul(self.matrix, point) for point in self.data1d]
        return transform.postprocess_data(self)

class rotate_transformation(transform):
    def __init__(self, data, degrees, visualize = False, center = None):
        transform.__init__(self, data, visualize)
        self.degrees = degrees * one_torch
        if center is None:
            self.center = torch.ones(1, 2)
            self.center[..., 0] = data['data2d'].shape[-2] // 2  # x
            self.center[..., 1] = data['data2d'].shape[-1] // 2  # y
        else:
            self.center = center
        self.center.to(device)
    def __call__(self, *args, **kwargs):
        self.data2d = kornia.geometry.rotate(self.data2d, angle=self.degrees , center=self.center)
        matrix = (
            kornia.geometry.get_rotation_matrix2d(angle=self.degrees, center=self.center, scale=one_torch)).reshape(2,
                                                                                                                      3)
        if self.data1d:
            self.data1d = [torch.matmul(matrix, point) for point in self.data1d]
        return transform.postprocess_data(self)

class  scale_transformation(transform):
    def __init__(self, data, scale_factor, visualize = False, center = None):
        transform.__init__(self, data, visualize)
        self.scale_factor = (torch.ones(1)*scale_factor).to(device)
        if center is None:
            self.center = torch.ones(1, 2)
            self.center[..., 0] = data['data2d'].shape[-2] // 2  # x
            self.center[..., 1] = data['data2d'].shape[-1] // 2  # y
        else:
            self.center = center
        self.center.to(device)

    def get_scale_matrix(center, scale_factor):
        if isinstance(scale_factor,
                      float) or scale_factor.dim() == 1:  # si solo se proporciona un valor; se escala por igual en ambos ejes
            scale_factor = torch.ones(2).to(device) * scale_factor
        matrix = torch.zeros(2, 3).to(device)
        matrix[0, 0] = scale_factor[0]
        matrix[1, 1] = scale_factor[1]
        matrix[0, 2] = (-scale_factor[0] + 1) * center[0]
        matrix[1, 2] = (-scale_factor[1] + 1) * center[1]
        return matrix

    def __call__(self, *args, **kwargs):
        self.data2d = kornia.geometry.scale(self.data2d, scale_factor=self.scale_factor, center=self.center)
        matrix = self.get_scale_matrix(self.center, self.scale_factor)
        if self.data1d is not None:
            self.data1d = [torch.matmul(matrix, point) for point in self.data1d]

class translate_transformation(transform):
    def __init__(self, data, translation, visualize = False):
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
        if self.data1d is not None:
            self.data1d = [self.translate_point(point, self.translation) for point in self.data1d]

