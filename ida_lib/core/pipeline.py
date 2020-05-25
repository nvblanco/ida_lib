from typing import Union
import cv2
import torch

from ida_lib.core.pipeline_operations import *
from ida_lib.core.pipeline_functional import *


class pipeline(object):
    '''
    The pipeline object represents the pipeline with data transformation operations (pictures, points). When executed, on a batch of images,
    it applies the necessary transformations (being different on each image based on the probabilities of each operation included).

        Considerations:
            1)  The images must be of the same size, or the RESIZE operation must be included so that the transformations can be applied correctly
            2)  To run the pipeline, it accepts any type of input metadata named in the input dict. In particular it gives special treatment
                to data named as:
                    - Mask:     it is affected by geometric transformations and its output is discretized to values of 0-1
                    - Image:    affected by geometric and color transformations
                    - Keypoints: geometric transformations are applied to them as coordinates.
                    - Others:   any other metadata will not be transformed (example: 'tag', 'target'...)

        Example:

                pip = pipeline(resize = (25, 25),  pipeline_operations=(
                                        translate_pipeline(probability=0.5, translation=(3, 0.05)),
                                        vflip_pipeline(probability=0.5),
                                        hflip_pipeline(probability=0.5),
                                        contrast_pipeline(probability=0.5, contrast_factor=1),
                                        random_brightness_pipeline(probability=0.2, brightness_range=(1.5, 1.6)),
                                        random_scale_pipeline(probability=1, scale_range=(0.5, 1.5), center_desviation=20),
                                        random_rotate_pipeline(probability=0.2, degrees_range=(-50, 50), center_desviation=20))
                                        ))
    '''

    def __init__(self, pipeline_operations: list, resize: tuple = None, interpolation: str = 'bilinear',
                 padding_mode: str = 'zeros'):
        '''
        :param pipeline_operations: list of pipeline initialized operations (see pipeline_operations.py)
        :param resize: tuple of desired output size. Example (25,25)
        :param interpolation (str) :interpolation mode to calculate output values
                'bilinear' | 'nearest'. Default: 'bilinear'.
        :param padding_mode(str): padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
        '''
        self.color_ops, self.geom_ops, self.indep_ops = split_operations_by_type(pipeline_operations)
        self.geom_ops.reverse()  # to apply matrix multiplication in the user order
        self.info_data = None
        self.resize = resize
        self.interpolation = interpolation
        self.padding_mode = padding_mode

    def apply_geometry_transform_data2d(self, image: torch.tensor, matrix: torch.tensor) -> torch.tensor:
        '''
        (Private method) Applies the input transform to the image by the padding and interpolation mode configured on the pipeline
        :param image (torch.tensor)         : image to transform
        :param matrix (torch.tensor)        : transformation matrix that represent the operation to be applied
        :return (torch.tensor)              : transformed image
        '''
        return own_affine(image, matrix[:2, :], interpolation=self.interpolation, padding_mode=self.padding_mode)

    def apply_geometry_transform_discreted_data2d(self, image: torch.tensor, matrix: torch.tensor) -> torch.tensor:
        '''
        (Private method) Applies the input transform to the image by the padding mode configured on the pipeline and 'nearest' interpolation to preserve discrete values of segmaps or masks
        :param image (torch.tensor)         : image to transform
        :param matrix (torch.tensor)        : transformation matrix that represent the operation to be applied
        :return (torch.tensor)              : transformed image
        '''
        return own_affine(image, matrix[:2, :], interpolation='nearest', padding_mode=self.padding_mode)

    def apply_geometry_transform_points(self, points_matrix: torch.tensor, matrix: torch.tensor) -> torch.tensor:
        '''
        (Private method) Applies the input tranform to a matrix of points coordinates (matrix multiplication)
        :param points_matrix (torch.tensor) : matrix of points coordinates
        :param matrix (torch.tensor)        : transformation matrix that represent the operation to be applied
        :return (torch.tensor)              : matrix of trasnsformed points coordinates
        '''
        return torch.matmul(matrix, points_matrix)

    def get_data_types(self) -> tuple :
        ''' Returns the tuple of data types identified on the input data'''
        return self.info_data['present_types']


    def __call__(self, batch_data: Union[list, dict], visualize: bool = False) -> Union[dict, list]:
        '''
        Applies the transformations to the input image batch.
        *   If it is the first batch entered into the pipeline, the information about the type of input data
            is analyzed and the different pipeline parameters are set (size of the images, labels, bits per pixel..)
        :param batch_data (list)    : list of elements to be tranformed through the pipe
        :param visualize  (bool)    : it allows to display the web visualization tool of performed transformations
        :return: transformed batch
        '''

        if not isinstance(batch_data, list): batch_data = [batch_data]
        if visualize: original = [d.copy() for d in batch_data]  # copy the original batch to diplay on visualization
        self.process_data = []
        if self.resize is None:
            if self.info_data is None:  # First iteration to configure parameters and scan data info while the first item is being processed
                data = batch_data[0]
                batch_data = batch_data[1:]  # exclude the first item in the batch to be processed on the second loop
                '''set the color depth'''
                '''bpp = int( data['image'].dtype.name[4:])
                max = pow(2, bpp) - 1
                global pixel_value_range
                pixel_value_range = (0, max // 2, max)'''
                '''get compose color functions '''
                lut = get_compose_function(self.color_ops)
                data['image'] = cv2.LUT(data['image'], lut)
                for op in self.indep_ops: data['image'] = op._apply_to_image_if_probability(data['image'])
                p_data, self.info_data = preprocess_dict_data_and_data_info(data, self.interpolation)
                matrix = get_compose_matrix_and_configure_parameters(self.geom_ops, self.info_data)
                p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
                if self.info_data['contains_discrete_data']: p_data[
                    'data_2d_discreted'] = self.apply_geometry_transform_discreted_data2d(p_data['data_2d_discreted'],
                                                                                          matrix)
                if self.info_data['contains_keypoints']: p_data['points_matrix'] = self.apply_geometry_transform_points(
                    p_data['points_matrix'], matrix)
                self.process_data.append(p_data)

            for data in batch_data:
                lut = get_compose_function(self.color_ops)
                data['image'] = cv2.LUT(data['image'], lut)
                for op in self.indep_ops: data['image'] = op._apply_to_image_if_probability(data['image'])
                p_data = preprocess_dict_data(data, self.info_data)
                matrix = get_compose_matrix(self.geom_ops)
                p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
                if self.info_data['contains_discrete_data']: p_data[
                    'data_2d_discreted'] = self.apply_geometry_transform_discreted_data2d(p_data['data_2d_discreted'],
                                                                                          matrix)
                if self.info_data['contains_keypoints']: p_data['points_matrix'] = self.apply_geometry_transform_points(
                    p_data['points_matrix'], matrix)
                self.process_data.append(p_data)
        else:
            if self.info_data is None:  # First iteration to configure parameters and scan data info while the first item is being processed
                data = batch_data[0]
                batch_data = batch_data[1:]  # exclude the first item in the batch to be processed on the second loop
                lut = get_compose_function(self.color_ops)
                data['image'] = cv2.LUT(data['image'], lut)
                for op in self.indep_ops: data['image'] = op._apply_to_image_if_probability(data['image'])
                p_data, self.info_data = preprocess_dict_data_and_data_info_with_resize(data, new_size=self.resize,
                                                                                        interpolation=self.interpolation)
                matrix = get_compose_matrix_and_configure_parameters(self.geom_ops, self.info_data)
                p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
                if self.info_data['contains_discrete_data']: p_data[
                    'data_2d_discreted'] = self.apply_geometry_transform_discreted_data2d(p_data['data_2d_discreted'],
                                                                                          matrix)
                if self.info_data['contains_keypoints']: p_data['points_matrix'] = self.apply_geometry_transform_points(
                    p_data['points_matrix'], matrix)
                self.process_data.append(p_data)

            for data in batch_data:
                lut = get_compose_function(self.color_ops)
                data['image'] = cv2.LUT(data['image'], lut)
                for op in self.indep_ops: data['image'] = op._apply_to_image_if_probability(data['image'])
                p_data = preprocess_dict_data_with_resize(data, self.info_data)
                matrix = get_compose_matrix(self.geom_ops)
                p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
                if self.info_data['contains_discrete_data']: p_data[
                    'data_2d_discreted'] = self.apply_geometry_transform_discreted_data2d(p_data['data_2d_discreted'],
                                                                                          matrix)
                if self.info_data['contains_keypoints']: p_data['points_matrix'] = self.apply_geometry_transform_points(
                    p_data['points_matrix'], matrix)
                self.process_data.append(p_data)
        if visualize:
            return postprocess_data_and_visualize(self.process_data, original, self.info_data)
        else:
            return postprocess_data(self.process_data, self.info_data)

