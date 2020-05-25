from functools import wraps
from typing import Union

import kornia
from string import digits
import torch
from image_augmentation import visualization
from . import utils

device = utils.device
data_types_2d = {"image", "mask", "heatmap"}
data_types_1d = {"keypoints"}

one_torch = torch.ones(1).to(device)


def prepare_data(func):
    '''
    Decorator that prepares the input data to apply the geometric transformation. For this purpose, it concatenates all
    the two-dimensional elements of the input data in the same tensor on which a single transformation is applied.  If
    the input data contains point coordinates, they are grouped in a matrix as homogeneous coordinates, over which a
    single matrix multiplication is performed.

    :param func: geometric function to be applied to the data
    :return: processed data
    '''

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
        if data_output.keys().__contains__('types_2d'):
            data_process = {}
            data_split = torch.split(data_output['data2d'], list(data_output['types_2d'].values()), dim=0)
            for index, type in enumerate(data_output['types_2d']):
                data_process[type] = data_split[index]
            if data_process.keys().__contains__('mask'): data_process['mask'] = utils.mask_change_to_01_functional(
                data_process['mask'])
            if data_output.keys().__contains__('points_matrix'): data_process['keypoints'] = [
                ((dato.cpu())[:2, :]).reshape(2) for dato in torch.split(data_output['points_matrix'], 1, dim=1)]
        else:
            data_process = data_output['data2d']
        if visualize:
            visualization.plot_image_tranformation(data_process, data_output['original'])
        return data_process

    return wrapped_function


'''---Vertical Flip---'''
def vflip_image(img: torch.tensor)-> torch.tensor:
    return kornia.vflip(img)

def vflip_coordiantes_matrix(matrix: torch.tensor, heigth: int)-> torch.tensor:
    matrix[1] = torch.ones(1, matrix.shape[1]).to(device) * (heigth) - \
    matrix[1]
    return matrix

@prepare_data
def vflip_compose_data(data: dict)->dict:
    '''
    :param data (dict) : dict of elements to be transformed
    :return: transformed data
    '''
    data['data2d'] = vflip_image(data['data2d'])
    heigth = data['data2d'].shape[-2]
    if data.keys().__contains__('points_matrix'):
        data['points_matrix'] = vflip_coordiantes_matrix(data['points_matrix'], heigth)
    return data

'''--- Horizontal Flip---'''

def hflip_image(img: torch.tensor)-> torch.tensor:
    return kornia.hflip(img)

def hflip_coordinates_matrix(matrix: torch.tensor, width: int)-> torch.tensor:
    matrix[0] = torch.ones(1, matrix.shape[1]).to(device) * (width) - \
                               matrix[0]
    return matrix

@prepare_data
def hflip_compose_data(data: dict) -> dict:
    '''
    :param data (dict) : dict of elements to be transformed
    :return: transformed data
    '''
    data['data2d'] = hflip_image(data['data2d'])
    width = data['data2d'].shape[-1]
    if data.keys().__contains__('points_matrix'):
        data['points_matrix'] = hflip_coordinates_matrix(data['points_matrix'], width)
    return data

''' --- Afine transform ---'''

def affine_image(img: torch.tensor, matrix: torch.tensor)-> torch.tensor:
    return  kornia.geometry.affine(img, matrix)

def affine_coordinates_matrix(matrix_coordinates: torch.tensor, matrix_transformation: torch.tensor) -> torch.tensor:
    return torch.matmul(matrix_transformation, matrix_coordinates)

@prepare_data
def affine_compose_data(data: dict, matrix: torch.tensor) -> dict:
    '''
    :param data     (dict)          : dict of elements to be transformed
    :param matrix   (torch.tensor)  : matrix of transformation
    :return: transformed data
    '''
    matrix = matrix.to(device)
    data['data2d'] = affine_image(data['data2d'], matrix)
    if data.keys().__contains__('points_matrix'):
        data['points_matrix'] = affine_coordinates_matrix(data['points_matrix'], matrix)
    return data

''' --- Rotate transform --- '''
def get_rotation_matrix(center: torch.tensor, degrees: torch.tensor):
    return ( kornia.geometry.get_rotation_matrix2d(angle=degrees, center=center, scale=one_torch)).reshape(2, 3)

def rotate_image(img: torch.tensor, degrees: torch.tensor, center: torch.tensor)-> torch.tensor:
    return  kornia.geometry.rotate(img, angle=degrees, center=center)

def rotate_coordinates_matrix(matrix_coordinates: torch.tensor, matrix: torch.tensor)-> torch.tensor:
    return torch.matmul(matrix, matrix_coordinates)

@prepare_data
def rotate_compose_data(data: dict, degrees: torch.tensor, center: torch.tensor):
    '''
    :param data     (dict)          : dict of elements to be transformed
    :param degrees  (torch.tensor)  : counterclockwise degrees of rotation
    :param center   (torch.tensor)  : center of rotation. Default, center of the image
    :return: transformed data
    '''
    degrees = degrees * one_torch
    if center is None:
        center = utils.get_torch_image_center(data['data2d'])
    else:
        center = center
    center = center.to(device)
    data['data2d'] = rotate_image(data['data2d'], degrees, center)
    matrix = get_rotation_matrix(center, degrees)
    if data.keys().__contains__('points_matrix'):
        data['points_matrix'] = rotate_coordinates_matrix(data['points_matrix'], matrix)
    return data

''' ---Scale Transform----'''
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
    '''
    :param data         (dict)          : dict of elements to be transformed
    :param scale_factor (float)         : factor of scaling
    :param center       (torch tensor)  : center of scaling. By default its taken the center of the image
    :return: transformed data
    '''
    scale_factor = (torch.ones(1) * scale_factor).to(device)
    if center is None:
        center = utils.get_torch_image_center(data['data2d'])
    center = center.to(device)
    data['data2d'] = scale_image(data['data2d'], scale_factor, center)
    matrix = get_scale_matrix(center, scale_factor)
    if data.keys().__contains__('points_matrix'):
        data['points_matrix'] = scale_coordinates_matrix( data['points_matrix'], matrix)
    return data

''' --- Translation transform ---'''
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
    '''
    :param data         (dict)          : dict of elements to be transformed
    :param translation  (torch.tensor)  : number of pixels to translate
    :return: transformed data
    '''
    if not torch.is_tensor(translation):
        translation = (torch.tensor(translation).float().reshape((1, 2)))
    translation = translation.to(device)
    data['data2d'] = translate_image(data['data2d'], translation)
    if data.keys().__contains__('points_matrix'):
        data['points_matrix'] =  translate_coordinates_matrix(data['points_matrix'], translation)
    return data


''' --- Shear Transform ---'''
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
    '''

    :param data         (dict)          : dict of elements to be transformed
    :param shear_factor (torch.tensor)  : pixels of shearing
    :return:
    '''
    shear_factor = (torch.tensor(shear_factor).reshape(1,2)).to(device)
    matrix = get_shear_matrix(shear_factor)
    data['data2d'] = shear_image(data['data2d'], matrix)
    matrix = get_shear_matrix(shear_factor)
    if data.keys().__contains__('points_matrix'):
        data['points_matrix'] =  shear_coordinates_matrix(data['points_matrix'], matrix)
    return data



'''
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
                self.data1d=None
                self.points_matrix=None
                self.types_2d = {}
                self.types_2d['image']=data.shape[0]

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
'''