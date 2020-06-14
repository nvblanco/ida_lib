import numpy as np
import pytest
import torch

from ida_lib.core.pipeline import Pipeline
from ida_lib.core.pipeline_geometric_ops import ScalePipeline, ShearPipeline, RandomScalePipeline, \
    RandomShearPipeline, TranslatePipeline, RandomTranslatePipeline, HflipPipeline, RandomRotatePipeline,\
    VflipPipeline, RotatePipeline
from ida_lib.core.pipeline_local_ops import GaussianNoisePipeline, BlurPipeline, GaussianBlurPipeline, \
    PoissonNoisePipeline, SaltAndPepperNoisePipeline, SpekleNoisePipeline
from ida_lib.core.pipeline_pixel_ops import ContrastPipeline, BrightnessPipeline, RandomContrastPipeline, \
    RandomBrightnessPipeline, GammaPipeline, RandomGammaPipeline, DenormalizePipeline, NormalizePipeline
from ida_lib.operations.utils import arrays_equal


# cp-009
def test_pipeline_resize(numpy_image_batch):
    shape = (10, 20)
    pip = Pipeline(interpolation='nearest',
                   resize=shape,
                   pipeline_operations=(
                       ScalePipeline(probability=0, scale_factor=0.5),
                       ShearPipeline(probability=0, shear=(0.2, 0.2)),
                       GaussianNoisePipeline(probability=0)))

    augmented = pip(numpy_image_batch)
    assert augmented[0]['image'].shape[1] == shape[0]
    assert augmented[0]['image'].shape[2] == shape[1]


# cp-010
def test_pipeline_int_numpy_image_pipeline(numpy_image_batch, pipeline):
    augmented = pipeline(numpy_image_batch)
    assert augmented[0]['image'].dtype == torch.uint8


# cp-011
def test_pipeline_int_numpy_image_empty_pipeline(numpy_image_batch, empty_pipeline):
    augmented = empty_pipeline(numpy_image_batch)
    assert augmented[0]['image'].dtype == torch.uint8


# cp-012
def test_pipeline_input_element_without_image(numpy_batch_without_image, pipeline):
    pipeline(numpy_batch_without_image)
    assert True


# cp-013
def test_pipeline_all_operations(numpy_image_batch):
    pip = Pipeline(interpolation='nearest',
                   pipeline_operations=(
                       ScalePipeline(probability=1, scale_factor=0.5),
                       RandomScalePipeline(probability=1, scale_range=(0.5, 1.5)),
                       ShearPipeline(probability=1, shear=(0.2, 0.2)),
                       RandomShearPipeline(probability=1, shear_range=(0, 0.2)),
                       TranslatePipeline(probability=1, translation=(10, 50)),
                       RandomTranslatePipeline(probability=1, translation_range=(10, 20)),
                       HflipPipeline(probability=1, exchange_points=[(0, 5), (1, 6)]),
                       RandomRotatePipeline(probability=1, degrees_range=(-20, 20)),
                       HflipPipeline(probability=1),
                       VflipPipeline(probability=1),
                       RotatePipeline(probability=1, degrees=0),
                       GaussianNoisePipeline(probability=1),
                       BlurPipeline(probability=1),
                       GaussianBlurPipeline(probability=1),
                       GaussianNoisePipeline(probability=1),
                       PoissonNoisePipeline(probability=1),
                       SaltAndPepperNoisePipeline(probability=1),
                       SpekleNoisePipeline(probability=1),
                       ContrastPipeline(probability=1, contrast_factor=1),
                       RandomContrastPipeline(probability=1, contrast_range=(0.9, 1.1)),
                       BrightnessPipeline(probability=1, brightness_factor=1),
                       RandomBrightnessPipeline(probability=1, brightness_range=(1, 1.3)),
                       GammaPipeline(probability=1, gamma_factor=1),
                       RandomGammaPipeline(probability=1, gamma_range=(1, 2)),
                       NormalizePipeline(probability=1),
                       DenormalizePipeline(probability=1)
                   ))
    pip(numpy_image_batch)
    assert True


# cp-014
def test_pipeline_switch_points(numpy_image_and_points_batch):
    original_points = numpy_image_and_points_batch[0]['keypoints'].copy()
    pip = Pipeline(interpolation='nearest',
                   pipeline_operations=(
                       HflipPipeline(probability=1, exchange_points=((0, 1), (3, 4))),
                       HflipPipeline(probability=1)))
    augmented = pip(numpy_image_and_points_batch)
    transformed_points = augmented[0]['keypoints'].cpu().numpy().astype(np.uint8)
    assert arrays_equal(original_points[0], transformed_points[1])
    assert arrays_equal(original_points[1], transformed_points[0])


# cp-015
def test_pipeline_all_element_float(pipeline, numpy_float_all_elements_batch):
    augmented = pipeline(numpy_float_all_elements_batch)
    assert augmented[0]['image'].dtype == torch.float64
    assert augmented[0]['mask'].dtype == torch.float64
    assert augmented[0]['segmap'].dtype == torch.float64
    assert augmented[0]['heatmap'].dtype == torch.float64


# cp-016
def test_pipeline_2_mask_float(pipeline, numpy_batch_2_mask):
    augmented = pipeline(numpy_batch_2_mask)
    assert augmented[0]['image'].dtype == torch.float64
    assert augmented[0]['mask'].dtype == torch.float64
    assert augmented[0]['mask2'].dtype == torch.float64


# cp-017
@pytest.mark.parametrize(
    ["params"], [[{'interpolation': 'nearest', 'padding_mode': 'zeros', 'output_format': 'dict'}],
                 [{'interpolation': 'bilinear', 'padding_mode': 'reflection', 'output_format': 'tuple'}],
                 [{'interpolation': 'nearest', 'padding_mode': 'border', 'output_format': 'dict'}]]
)
def test_interpolation_valid_inputs(numpy_image_batch, params):
    pip = Pipeline(**params,
                   pipeline_operations=(
                       HflipPipeline(probability=1, exchange_points=((0, 1), (3, 4))),
                       HflipPipeline(probability=1))
                   )
    pip(numpy_image_batch)
    assert True


# cp-018
@pytest.mark.parametrize(
    ["params"], [[{'interpolation': 'invalid', 'padding_mode': 'zeros', 'output_format': 'dict'}],
                 [{'interpolation': 'bilinear', 'padding_mode': 'invalid', 'output_format': 'tuple'}],
                 [{'interpolation': 'nearest', 'padding_mode': 'border', 'output_format': 'invalid'}]]
)
def test_interpolation_invalid_inputs(numpy_image_batch, params):
    try:
        pip = Pipeline(**params,
                       pipeline_operations=(
                           HflipPipeline(probability=1, exchange_points=((0, 1), (3, 4))),
                           HflipPipeline(probability=1))
                       )
        pip(numpy_image_batch)
        assert False
    except ValueError:
        assert True


# cp-019
@pytest.mark.parametrize(
    ["output_type"], [[torch.uint8], [torch.int16], [torch.int32], [torch.int64], [torch.float16], [torch.float32],
                      [torch.float64]]
)
def test_pipeline_output_type(numpy_float_all_elements_batch, output_type):
    pip = Pipeline(interpolation='nearest',
                   output_type=output_type,
                   pipeline_operations=(
                       HflipPipeline(probability=1, exchange_points=((0, 1), (3, 4))),
                       HflipPipeline(probability=1)))
    augmented = pip(numpy_float_all_elements_batch)
    assert augmented[0]['image'].dtype == output_type
    assert augmented[0]['mask'].dtype == output_type
    assert augmented[0]['segmap'].dtype == output_type
    assert augmented[0]['heatmap'].dtype == output_type
