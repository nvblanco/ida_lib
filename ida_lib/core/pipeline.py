from typing import Union, Optional

import cv2
import torch

from ida_lib.core.pipeline_functional import (split_operations_by_type, get_compose_function,
                                              preprocess_data,
                                              get_compose_matrix,
                                              postprocess_data, switch_point_positions)
from ida_lib.operations.geometry_ops_functional import own_affine
from ida_lib.operations.utils import get_principal_type, dtype_to_torch_type, add_new_axis


class Pipeline(object):
    """
    The pipeline object represents the pipeline with data transformation operations (pictures, points). When executed,
    on a batch of images,it applies the necessary transformations (being different on each image based on the
    probabilities of each operation included).

        Considerations:
            1)  The images must be of the same size, or the RESIZE operation must be included so that the
            transformations can be applied correctly
            2)  To run the pipeline, it accepts any type of input metadata named in the input dict. In particular it
            gives special treatment
                to data named as:
                    - Mask: it is affected by geometric transformations and its output is discrete to values of 0-1
                    - Segmap: generalization of mask. Every value is discrete
                    - Image:  affected by geometric and color transformations
                    - Keypoints: geometric transformations are applied to them as coordinates.
                    - Others: any other metadata will not be transformed (example: 'tag', 'target'...)

        Example:

             pip = pipeline(resize = (25, 25),  pipeline_operations=(
                                translate_pipeline(probability=0.5, translation=(3, 0.05)),
                                vflip_pipeline(probability=0.5),
                                hflip_pipeline(probability=0.5),
                                contrast_pipeline(probability=0.5, contrast_factor=1),
                                random_brightness_pipeline(probability=0.2, brightness_range=(1.5, 1.6)),
                                random_scale_pipeline(probability=1, scale_range=(0.5, 1.5), center_deviation=20),
                                random_rotate_pipeline(probability=0.2, degrees_range=(-50, 50), center_deviation=20))
                                  ))
    """

    def __init__(self, pipeline_operations: list, resize: tuple = None, interpolation: str = 'bilinear',
                 padding_mode: str = 'zeros', output_format: str = 'dict',
                 output_type: Optional[torch.dtype] = None):
        """

        :param pipeline_operations: list of pipeline initialized operations (see pipeline_operations.py)
        :param resize: tuple of desired output size. Example (25,25)
        :param interpolation:interpolation mode to calculate output values
                'bilinear' | 'nearest'. Default: 'bilinear'.
        :param padding_mode: padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
        :param output_format: desired format for each output item in the pipeline
                'dict' (each item accompanied by its type name) | 'tuple'
        :param: output_type: desired type for the bidimensional output items. If it is None, output type will be the
        same as input's type

        """
        if interpolation not in ('bilinear', 'nearest'):
            raise ValueError('interpolation has to be "nearest" or "bilinear". Got ' + interpolation)
        if padding_mode not in ('zeros', 'border', 'reflection'):
            raise ValueError('padding_mode has to be "zeros", "border" or "reflection" . Got ' + padding_mode)
        if output_format not in ('dict', 'tuple'):
            raise ValueError('output_format has to be "dict" or "tuple" . Got ' + output_format)

        self.color_ops, self.geom_ops, self.indep_ops = split_operations_by_type(pipeline_operations)
        self.geom_ops.reverse()  # to apply matrix multiplication in the user order
        self.info_data = None
        self.resize = resize
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.output_format = output_format
        self.output_type = output_type

    def _apply_geometry_transform_data2d(self, image: torch.tensor, matrix: torch.tensor) -> torch.tensor:
        """
        Applies the input transform to the image by the padding and interpolation mode configured on the pipeline

        :param image: image to transform
        :param matrix: transformation matrix that represent the operation to be applied
        :return: transformed image
        """
        return own_affine(image, matrix[:2, :], interpolation=self.interpolation, padding_mode=self.padding_mode)

    def _apply_geometry_transform_discrete_data2d(self, image: torch.tensor, matrix: torch.tensor) -> torch.tensor:
        """
        Applies the input transform to the image by the padding mode configured on the pipeline and 'nearest'
        interpolation to preserve discrete values of segmaps or masks
        :param image: image to transform
        :param matrix: transformation matrix that represent the operation to be applied
        :return: transformed image
        """
        return own_affine(image, matrix[:2, :], interpolation='nearest', padding_mode=self.padding_mode)

    @staticmethod
    def _apply_geometry_transform_points(points_matrix: torch.tensor, matrix: torch.tensor) -> torch.tensor:
        """

        Applies the input transform to a matrix of points coordinates (matrix multiplication)
        :param points_matrix: matrix of points coordinates
        :param matrix: transformation matrix that represent the operation to be applied
        :return: matrix of transformed points coordinates
        """
        return torch.matmul(matrix, points_matrix)

    def get_data_types(self) -> tuple:
        """ Returns the tuple of data types identified on the input data"""
        return self.info_data['present_types']

    def __call__(self, batch_data: Union[list, dict], visualize: bool = False) -> Union[dict, list]:
        """
        Applies the transformations to the input image batch.
        *   If it is the first batch entered into the pipeline, the information about the type of input data
            is analyzed and the different pipeline parameters are set (size of the images, labels, bits per pixel..)

        :param batch_data: list of elements to be transformed through the pipe
        :param visualize: it allows to display the web visualization tool of performed transformations
        :return:  transformed batch
        """

        # Configure data and initial parameters
        if not isinstance(batch_data, list):
            batch_data = [batch_data]
        original = None
        # a copy of the original data is made for display (if visualize is True)
        if visualize:
            original = [d.copy() for d in batch_data]  # copy the original batch to display on visualization
        self.process_data = []
        principal_type = get_principal_type(batch_data[0])
        if self.output_type is None:
            self.output_type = dtype_to_torch_type(batch_data[0][principal_type].dtype)

        # Start looping over the batch items
        for index, data in enumerate(batch_data):
            # only perform color operations if the item contains any image
            if 'image' in data:
                lut = get_compose_function(self.color_ops)
                data['image'] = cv2.LUT(data['image'].astype('uint8'), lut)
                if len(data['image'].shape) == 2:
                    data['image'] = add_new_axis(data['image'])
                for op in self.indep_ops:
                    data['image'] = op.apply_to_image_if_probability(data['image'])
            # process data and get compose matrix of geometric  operations
            if self.info_data is None:  # Needs to configure batch information
                p_data, self.info_data = preprocess_data(data, interpolation=self.interpolation, resize=self.resize)
                matrix, switch_points = get_compose_matrix(self.geom_ops, self.info_data)  # calculates the composite
                # matrix and configures the necessary parameters (causes by batch_info as a parameter)

            else:  # Batch information has already been set up
                p_data = preprocess_data(data, batch_info=self.info_data, resize=self.resize)
                matrix, switch_points = get_compose_matrix(self.geom_ops)

            # perform the geometry  compose transform
            p_data['data_2d'] = self._apply_geometry_transform_data2d(p_data['data_2d'], matrix)

            if self.info_data['contains_discrete_data']:
                # if there are segmaps or masks, the transformations are applied to them with discrete values
                p_data['data_2d_discrete'] = self._apply_geometry_transform_discrete_data2d(
                    p_data['data_2d_discrete'], matrix)
            if self.info_data['contains_keypoints']:
                if switch_points:  # if necessary, the order of the points is changed
                    switch_point_positions(p_data['points_matrix'], switch_points)
                p_data['points_matrix'] = self._apply_geometry_transform_points(p_data['points_matrix'], matrix)
            self.process_data.append(p_data)

        # Once all the elements of the batch have been transformed, they are restructured for the output
        return postprocess_data(batch=self.process_data, batch_info=self.info_data, data_original=original,
                                visualize=visualize, original_type=self.output_type, output_format=self.output_format)
