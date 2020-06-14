import numpy as np
import pytest
import torch

from ida_lib.global_parameters import identity
from ida_lib.operations.transforms import (hflip,
                                           vflip,
                                           affine,
                                           rotate,
                                           shear,
                                           scale,
                                           translate,
                                           change_gamma,
                                           change_contrast,
                                           change_brightness,
                                           equalize_histogram,
                                           inject_gaussian_noise,
                                           inject_poisson_noise,
                                           inject_spekle_noise,
                                           inject_salt_and_pepper_noise,
                                           blur,
                                           gaussian_blur)

identity_operations_and_params = [
    [hflip, {}],
    [vflip, {}],
    [rotate, {'degrees': 0}],
    [affine, {'matrix': identity}],
    [shear, {'shear_factor': (0, 0)}],
    [scale, {'scale_factor': 1}],
    [translate, {'translation': (0, 0)}],
    [change_gamma, {'gamma': 0}],
    [change_contrast, {'contrast': 1}],
    [change_brightness, {'bright': 1}],
    [inject_gaussian_noise, {}],
    [equalize_histogram, {}],
    [inject_spekle_noise, {}],
    [inject_poisson_noise, {}],
    [inject_salt_and_pepper_noise, {}],
    [blur, {}],
    [gaussian_blur, {}]]


# cp-001
@pytest.mark.parametrize(
    ["augmentation", "params"], identity_operations_and_params
)
def test_int_numpy_image_operations(augmentation, params, numpy_image):
    augmented = augmentation(numpy_image, **params)
    assert augmented['image'].dtype == np.uint8

    assert augmented['image'].shape[2] == 3


# cp-002
@pytest.mark.parametrize(
    ["augmentation", "params"], identity_operations_and_params
)
def test_int_numpy__monochannel_image_operations(augmentation, params, numpy_monochannel_image):
    augmented = augmentation(numpy_monochannel_image, **params)
    assert augmented['image'].dtype == np.uint8
    print(augmented['image'].shape)
    assert augmented['image'].shape[2] == 1


# cp-003
@pytest.mark.parametrize(
    ["augmentation", "params"], identity_operations_and_params
)
def test_int_tensor_image_operations(augmentation, params, torch_image):
    augmented = augmentation(torch_image, **params)
    assert augmented['image'].dtype == torch.uint8


# cp-004
@pytest.mark.parametrize(
    ["augmentation", "params"], identity_operations_and_params
)
def test_float_numpy_combined_operations(augmentation, params, numpy_float_all_elements_item):
    augmented = augmentation(numpy_float_all_elements_item, **params)
    assert augmented['image'].dtype == np.float
    assert augmented['mask'].dtype == np.float
    assert augmented['segmap'].dtype == np.float
    assert augmented['heatmap'].dtype == np.float


# cp-005
@pytest.mark.parametrize(
    ["augmentation", "params"], identity_operations_and_params
)
def test_float_tensor_combined_operations(augmentation, params, torch_float_all_elements_item):
    augmented = augmentation(torch_float_all_elements_item, **params)
    assert augmented['image'].dtype == torch.float64
    assert augmented['mask'].dtype == torch.float64
    assert augmented['segmap'].dtype == torch.float64
    assert augmented['heatmap'].dtype == torch.float64


identity_pixel_operations_and_params = [
    [change_gamma, {'gamma': 0}],
    [change_contrast, {'contrast': 1}],
    [change_brightness, {'bright': 1}],
    [inject_gaussian_noise, {}],
    [equalize_histogram, {}],
    [inject_spekle_noise, {}],
    [inject_poisson_noise, {}],
    [inject_salt_and_pepper_noise, {}],
    [blur, {}],
    [gaussian_blur, {}]]


# cp-006
@pytest.mark.parametrize(
    ["augmentation", "params"], identity_pixel_operations_and_params
)
# Assert the pixel operations without image raises exception
def test_operations_without_image(augmentation, params, numpy_item_without_image):
    try:
        augmentation(numpy_item_without_image, **params)
        assert False
    except AttributeError:
        assert True


@pytest.mark.parametrize(
    ["augmentation", "params"], identity_pixel_operations_and_params
)
# Assert the pixel operations without image raises exception
def test_operations_with_2_masks(augmentation, params, numpy_item_2_mask):
    augmented = augmentation(numpy_item_2_mask, **params)
    assert augmented['image'].dtype == np.float
    assert augmented['mask'].dtype == np.float
    assert augmented['mask2'].dtype == np.float


identity_pixel_operations_and_params = [
    [hflip, {}],
    [vflip, {}],
    [rotate, {'degrees': 0}],
    [affine, {'matrix': identity}],
    [shear, {'shear_factor': (0, 0)}],
    [scale, {'scale_factor': 1}],
    [translate, {'translation': (0, 0)}]
]


@pytest.mark.parametrize(
    ["augmentation", "params"], identity_pixel_operations_and_params
)
# Assert the pixel operations without image raises exception
def test_operations_without_image(augmentation, params, numpy_item_without_image):
    augmentation(numpy_item_without_image, **params)
    assert True
