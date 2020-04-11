import kornia
import torch

from . import functional

data_types_2d = {"image", "mask", "heatmap"}
data_types_1d = {"keypoints"}

device = 'cuda'
one_torch = torch.ones(1).to(device)

'''
DATA TYPES:
    * image:    torch tensor (C, H, W)
    * mask:     torch tensor (C, H, W)
    * hetamaps: torch tensor (C, H, W)
    
    * keypoints:list of torch tensor of dims (H, W) --keypoints must be with some 2d data'''

def hflip(data, visualize = False):
    op = functional.hflip_transformation(data, visualize)
    return op()
    '''
    data = preprocess_data(data, visualize )
    data['data2d'] = kornia.hflip(data['data2d'])
    if data.keys().__contains__('data1d'):
        width = data['data2d'].shape[-1]
        for point in data['data1d']:
            point[0] = width - point[0]
    return postprocess_data(data, visualize)
    '''


def vflip(data, visualize = False):
    op = functional.vflip_transformation(data, visualize)
    return op()
    '''
    data = preprocess_data(data, visualize)
    data['data2d'] = kornia.vflip(data['data2d'])
    if data.keys().__contains__('data1d'):
        heigth = data['data2d'].shape[-2]
        for point in data['data1d']:
            point[1] = heigth - point[1]
    return postprocess_data(data, visualize)'''

def affine(data, matrix, visualize = False):
    op = functional.affine_transformation(data,matrix, visualize)
    return op()
    '''
    data = preprocess_data(data, visualize)
    matrix = matrix.to(device)
    kornia.geometry.affine(data['data2d'], matrix)
    if data.keys().__contains__('data1d'):
        data['data1d'] = [torch.matmul(matrix, point) for point in data['data1d']]
        #data['data1d'] = [matrix * point for point in data['data1d']]
    return postprocess_data(data, visualize)'''

def rotate(data, degrees, visualize = False, center = None):
    if center is None:
        center = torch.ones(1, 2)
        center[..., 0] = data['data2d'].shape[-2] // 2  # x
        center[..., 1] = data['data2d'].shape[-1] // 2 # y
    center = center.to(device)
    data = preprocess_data(data, visualize)
    data['data2d'] = kornia.geometry.rotate(data['data2d'], angle=degrees*one_torch, center=center)
    matrix = (kornia.geometry.get_rotation_matrix2d(angle=one_torch * degrees, center = center, scale = one_torch)).reshape(2,3)
    if data.keys().__contains__('data1d'):
        data['data1d'] = [torch.matmul(matrix, point) for point in data['data1d']]
    return postprocess_data(data, visualize)


def x_shear_point(point, shear_factor):
    point[0] = point[0] + point[1]*shear_factor
    return point

def x_shear(data, shear_factor, visualize = False):
    data = preprocess_data(data, visualize)
    shear_factor_m = torch.zeros((1,2)).to(device)
    shear_factor_m[:,0]= shear_factor
    data['data2d'] = kornia.geometry.shear(data['data2d'], shear_factor_m)
    if data.keys().__contains__('data1d'):
        data['data1d'] = [x_shear_point(point, shear_factor)for point in data['data1d']]
    return postprocess_data(data, visualize)


def y_shear_point(point, shear_factor, visualize = False):
    point[1] = point[1] + point[0]*shear_factor
    return point

def y_shear(data, shear_factor, visualize = False):
    data = preprocess_data(data, visualize)
    shear_factor_m = torch.zeros((1, 2)).to(device)
    shear_factor_m[:, 1] = shear_factor
    data['data2d'] = kornia.geometry.shear(data['data2d'], shear_factor_m)
    if data.keys().__contains__('data1d'):
        data['data1d'] = [y_shear_point(point, shear_factor)for point in data['data1d']]
    return postprocess_data(data, visualize)


def get_scale_matrix(center, scale_factor):
    if isinstance(scale_factor, float) or scale_factor.dim()== 1: #si solo se proporciona un valor; se escala por igual en ambos ejes
        scale_factor = torch.ones(2).to(device) * scale_factor
    matrix = torch.zeros(2,3).to(device)
    matrix[0,0]=scale_factor[0]
    matrix[1,1]=scale_factor[1]
    matrix[0,2] = (-scale_factor[0] + 1)*center[0]
    matrix[1,2] = (-scale_factor[1] + 1)*center[1]
    return matrix

def scale(data, scale_factor, visualize = False, center = None):
    data = preprocess_data(data, visualize)
    if center is None:
        center = torch.ones(2)
        center[0] = data['data2d'].shape[-2] // 2  # x
        center[1] = data['data2d'].shape[-1] // 2 # y
    center = center.to(device)
    scale_factor = (torch.ones(1)*scale_factor).to(device)
    data['data2d'] = kornia.geometry.scale(data['data2d'], scale_factor=scale_factor , center=center)
    matrix = get_scale_matrix(center, scale_factor)
    if data.keys().__contains__('data1d'):
        data['data1d'] = [torch.matmul(matrix, point) for point in data['data1d']]
    return postprocess_data(data, visualize)



def translate_point(point, translation):
    point[0] += translation[:,0]
    point[1] += translation[:,1]
    return point

def translate(data, translation, visualize = False):
    if not torch.is_tensor(translation, visualize):
        translation = (torch.tensor(translation).float().reshape((1,2)))
    translation  = translation.to(device)
    data = preprocess_data(data)
    data['data2d'] = kornia.geometry.translate(data['data2d'], translation)
    if data.keys().__contains__('data1d'):
        data['data1d'] = [translate_point(point, translation) for point in data['data1d']]
    return postprocess_data(data, visualize)







