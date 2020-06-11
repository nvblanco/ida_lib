from string import digits

import cv2
import kornia
import numpy as np
import torch

from ida_lib.global_parameters import device, data_types_2d


def data_to_numpy(data):
    if torch.is_tensor(data):
        return kornia.tensor_to_image(data)
    elif isinstance(data, dict):
        for k in data.keys():
            if torch.is_tensor(data[k]):
                data[k] = kornia.tensor_to_image(data[k])
        return data
    else:
        return data


def round_torch(arr: torch.tensor, n_digits: int = 3):
    return torch.round(arr * 10 ** n_digits) / (10 ** n_digits)


def remove_digits(label: str):
    remove = str.maketrans('', '', digits)
    return label.translate(remove)


def add_new_axis(arr: np.ndarray):
    return arr[..., np.newaxis]


def dtype_to_torch_type(im_type: np.dtype):
    """
    Maps the numpy type to the equivalent torch.type
    :param im_type: numpy type
    :return: torch.type
    """
    if im_type == np.dtype('uint8'):
        return torch.uint8
    elif im_type == np.dtype('int8'):
        return torch.int8
    elif im_type == np.dtype('int16'):
        return torch.int16
    elif im_type == np.dtype('int32'):
        return torch.int
    elif im_type == np.dtype('int64'):
        return torch.int64
    elif im_type == np.dtype('float32'):
        return torch.float
    elif im_type == np.dtype('float64'):
        return torch.float64
    else:
        return torch.uint8


def get_principal_type(data: dict):
    if 'image' in data:
        return 'image'
    for label in data.keys():
        no_numbered = remove_digits(label)
        if no_numbered in data_types_2d:
            return label


def save_im(tensor, title):
    tensor = tensor.cpu()
    img = kornia.tensor_to_image(tensor.byte())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(title, img)


def element_to_dict_csv_format(item, name):
    output_dict = {}
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for index, item in enumerate(item):
            label = name + '_' + str(index)
            if isinstance(item, list) or isinstance(item, np.ndarray):
                labelx = label + '_x'
                labely = label + '_y'
                output_dict[labelx] = item[0]
                output_dict[labely] = item[1]
            else:
                output_dict[label] = item
    else:
        output_dict[name] = item
    return output_dict


"""Returns a tensor (two-dimensional) of the coordinates of the center of the input image """


def get_torch_image_center(data):
    center = torch.ones(1, 2)
    center[..., 0] = data.shape[-2] // 2  # x
    center[..., 1] = data.shape[-1] // 2  # y
    return center


def map_value(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def keypoints_to_homogeneus_functional(keypoints):
    if keypoints[0].dim() == 1:
        keypoints = [point.reshape(2, 1) for point in keypoints]
    return tuple([torch.cat((point.float(), torch.ones(1, 1)), axis=0).to(device) for point in keypoints])


def homogeneus_points_to_matrix(keypoints):
    return torch.transpose(keypoints[:2, :], 0, 1)


def homogeneus_points_to_list(keypoints):
    return [(dato[:2, :]).reshape(2) for dato in torch.split(keypoints, 1, dim=1)]


def keypoints_to_homogeneus_and_concatenate(keypoints, resize_factor=None):
    if resize_factor is None:
        if type(keypoints) is np.ndarray:
            keypoints = keypoints.transpose()
            ones = np.ones((1, keypoints.shape[1]))
            compose_data = torch.tensor(np.concatenate((keypoints, ones), axis=0), dtype=torch.float).to(device)
        else:
            if keypoints[0].dim() == 1:
                keypoints = [point.reshape(2, 1) for point in keypoints]
            keypoints = tuple([torch.cat((point.float(), torch.ones(1, 1)), axis=0).to(device) for point in keypoints])
            compose_data = torch.cat(keypoints, 1)  # concatenate data into one multichannel pytoch tensor
        return compose_data
    else:
        if type(keypoints) is np.ndarray:
            keypoints = keypoints.transpose()
            ones = np.ones((1, keypoints.shape[1]))
            compose_data = torch.tensor(np.concatenate(((keypoints[0, :] * resize_factor[0]).reshape(1, keypoints.shape[
                1]), (keypoints[1, :] * resize_factor[1]).reshape(1, keypoints.shape[1]), ones), axis=0),
                                        dtype=torch.float).to(device)
        else:
            if keypoints[0].dim() == 1:
                keypoints = [point.reshape(2, 1) for point in keypoints]
            keypoints = tuple([(torch.cat((torch.tensor(
                (point[0].float() * resize_factor[0], point[1].float() * resize_factor[1])).reshape(2, 1),
                                           torch.ones(1, 1)), axis=0).to(device)) for point in keypoints])
            compose_data = torch.cat(keypoints, 1)  # concatenate data into one multichannel pytoch tensor
        return compose_data


# converts the intermediate values ​​generated by the transformations to 0-1
def mask_change_to_01_functional(mask):
    return mask // 0.5


def _resize_image(image, new_size):
    return cv2.resize(image, new_size)


def is_a_normalized_image(image):
    return image.min() >= 0 and image.max() <= 1
