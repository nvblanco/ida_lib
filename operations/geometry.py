from . import geometry_functional

'''
DATA TYPES:
    * image:    torch tensor (C, H, W)
    * mask:     torch tensor (C, H, W)
    * hetamaps: torch tensor (C, H, W)
    
    * keypoints:list of torch tensor of dims (H, W) --keypoints must be with some 2d data'''

def hflip(data, visualize = False):
    op = geometry_functional.hflip_transformation(data, visualize)
    return op()


def vflip(data, visualize = False):
    op = geometry_functional.vflip_transformation(data, visualize)
    return op()

def affine(data, matrix, visualize = False):
    op = geometry_functional.affine_transformation(data, matrix, visualize)
    return op()


def rotate(data, degrees, visualize = False, center = None):
    op = geometry_functional.rotate_transformation(data, degrees, visualize, center)
    return op()

def scale(data, scale_factor, visualize = False, center = None):
    op = geometry_functional.rotate_transformation(data, scale_factor, visualize, center)
    return op()

def translate(data, translation, visualize = False):
    op = geometry_functional.translate_transformationns(data, translation, visualize)
    return op()

'''
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
    return postprocess_data(data, visualize)'''












