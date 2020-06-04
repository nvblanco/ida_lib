import kornia
import torch

device = 'cuda'


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
    if keypoints[0].dim() == 1: keypoints = [point.reshape(2, 1) for point in keypoints]
    return tuple([torch.cat((point.float(), torch.ones(1, 1)), axis=0).to(device) for point in keypoints])

def homogeneus_points_to_matrix(keypoints):
    return torch.transpose(keypoints[:2,:], 0, 1)
    """matrix = kornia.tensor_to_image(keypoints)
    return (matrix[:2, :]).transpose()"""

def homogeneus_points_to_list(keypoints):
    return [((dato)[:2, :]).reshape(2) for dato in torch.split(keypoints, 1, dim=1)]


def keypoints_to_homogeneus_and_concatenate(keypoints):
    if type(keypoints) is np.ndarray:
        keypoints = keypoints.transpose()
        ones = np.ones((1, keypoints.shape[1]))
        compose_data = torch.tensor(np.concatenate((keypoints, ones), axis=0), dtype=torch.float).to(device)
    else:
        if keypoints[0].dim() == 1: keypoints = [point.reshape(2, 1) for point in keypoints]
        keypoints = tuple([torch.cat((point.float(), torch.ones(1, 1)), axis=0).to(device) for point in keypoints])
        compose_data = torch.cat((keypoints), 1)  # concatenate data into one multichannel pytoch tensor
    return compose_data


def keypoints_to_homogeneus_and_concatenate_with_resize(keypoints, resize_factor):
    if type(keypoints) is np.ndarray:
        keypoints = keypoints.transpose()
        ones = np.ones((1, keypoints.shape[1]))
        compose_data = torch.tensor(np.concatenate(((keypoints[0,:] * resize_factor[0]).reshape(1,keypoints.shape[1]),(keypoints[1,:] * resize_factor[1]).reshape(1,keypoints.shape[1]),  ones), axis=0), dtype=torch.float).to(device)
    else:
        if keypoints[0].dim() == 1: keypoints = [point.reshape(2, 1) for point in keypoints]
        keypoints = tuple([(torch.cat((torch.tensor((point[0].float() * resize_factor[0] , point[1].float() * resize_factor[1])).reshape(2,1) , torch.ones(1, 1)), axis=0).to(device)) for point in keypoints])
        compose_data = torch.cat((keypoints), 1)  # concatenate data into one multichannel pytoch tensor
    return compose_data

# converts the intermediate values ​​generated by the transformations to 0-1
def mask_change_to_01_functional(mask):
    return (mask // 0.5)


import numpy as np
import os
import cv2

def _resize_image(image, new_size):
    return cv2.resize(image, new_size)

def _apply_gaussian_noise(image, var = 20):
    gaussian_noise = np.zeros((image.shape[0], image.shape[1],1), dtype=np.uint8)
    cv2.randn(gaussian_noise, 50, 20)
    gaussian_noise = np.concatenate((gaussian_noise, gaussian_noise, gaussian_noise), axis=2)
    gaussian_noise = (gaussian_noise * var).astype(np.uint8)
    return cv2.add(image, gaussian_noise)

def _apply_salt_and_pepper_noise(image, amount=0.05, s_vs_p = 0.5 ):
    if not is_a_normalized_image(image):
        salt = 255
    else:
        salt = 1
    pepper = 0
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords[0], coords[1], :] = salt
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords[0], coords[1], :] = pepper
    return out

def _apply_poisson_noise(image):
    noise = np.random.poisson(40, image.shape)
    return image + noise

def _apply_spekle_noise(image, mean=0, var=0.01):
    gauss = np.random.normal(mean, var ** 0.5, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + image * gauss
    return noisy

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def apply_gaussian_blur(img, blur_size=(5, 5)):
    return cv2.GaussianBlur(img, blur_size,cv2.BORDER_DEFAULT)

def _apply_blur(img,  blur_size=(5, 5)):
    return cv2.blur(img, (5,5))

"""source; https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv"""
def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.5
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

def is_a_normalized_image(image):
    return image.min() >= 0 and image.max() <=1


def is_color_image(image):
    return len(image.shape) == 3 and image.shape[2] == 3