import random
from typing import Union

import kornia
import torch

from ida_lib.core.pipeline_operations import PipelineOperation
from ida_lib.global_parameters import ones_torch, identity, one_torch, device, ones_2_torch
from ida_lib.operations.geometry_ops_functional import get_translation_matrix, get_squared_shear_matrix, \
    get_squared_scale_matrix

__all__ = ['HflipPipeline',
           'VflipPipeline',
           'RotatePipeline',
           'ShearPipeline',
           'ScalePipeline',
           'TranslatePipeline',
           'RandomScalePipeline',
           'RandomRotatePipeline',
           'RandomShearPipeline',
           'RandomTranslatePipeline']


class ScalePipeline(PipelineOperation):
    """Scale the input image-mask-keypoints and 2d data by the input scaling value"""

    def __init__(self, scale_factor: Union[float, tuple], probability: float = 1, center: torch.tensor = None):
        """
        :param scale_factor: scale value
        :param probability: probability of applying the transform. Default: 1.
        :param center: coordinates of the center of scaling. Default: center of the image
        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        self.__class__.__name__ = 'Scale pipeline'
        if center is None:
            self.config = True
            self.center = ones_torch.clone()
        else:
            self.center = center
            self.config = False
        self.matrix = identity.clone()
        self.ones_2 = ones_2_torch
        self.scale_factor = scale_factor

    def config_parameters(self, data_info: dict):
        self.config = False
        self.center[..., 1] = data_info['shape'][-2] // 2
        self.center[..., 0] = data_info['shape'][-3] // 2
        self.matrix = get_squared_scale_matrix(self.center, self.scale_factor)

    def get_op_matrix(self):
        return self.matrix

    def need_data_info(self):
        return self.config

    @staticmethod
    def switch_points():
        return None


class RandomScalePipeline(PipelineOperation):
    """Scale the input image-mask-keypoints and 2d data by a random scaling value calculated within the input range"""

    def __init__(self, probability: float, scale_range: tuple, keep_aspect: bool = True, center_deviation: int = None,
                 center: torch.tensor = None):
        """
        :param probability:[0-1] probability of applying the transform. Default: 1.
        :param scale_range: scale value
        :param keep_aspect: whether the scaling should be the same on the X axis and on the Y axis. Default: true
        :param center_deviation: produces random deviations at the scaling center. The deviations will be a maximum
                                  of the number of pixels indicated in this parameter
        :param center: coordinates of the center of scaling. Default: center of the image
        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        self.keep_aspect = keep_aspect
        if center is None:
            self.config = True
            self.center = ones_torch.clone()
        else:
            self.center = center
            self.config = False
        self.matrix = identity.clone()
        self.ones_2 = ones_2_torch
        self.scale_factor = self.ones_2
        if not isinstance(scale_range, tuple):
            raise Exception("Scale range must be a tuple")
        else:
            self.scale_factor[0] = one_torch * scale_range[0]
            self.scale_factor[1] = one_torch * scale_range[1]
        self.center_deviation = center_deviation

    def config_parameters(self, data_info: dict):
        self.config = False
        self.center[..., 1] = data_info['shape'][-2] // 2
        self.center[..., 0] = data_info['shape'][-3] // 2

    def get_op_matrix(self) -> torch.tensor:
        scale_factor_x = random.uniform(self.scale_factor[0], self.scale_factor[1])
        if self.keep_aspect:
            scale_factor_y = scale_factor_x
        else:
            scale_factor_y = random.uniform(self.scale_factor[0], self.scale_factor[1])
        self.matrix[0, 0] = scale_factor_x
        self.matrix[1, 1] = scale_factor_y
        if self.center_deviation is not None:
            self.center[:, 0] += random.randint(0, self.center_deviation)
            self.matrix[0, 2] = (-scale_factor_x + 1) * (
                    self.center[:, 0] + random.randint(-self.center_deviation, self.center_deviation))
            self.matrix[1, 2] = (-scale_factor_y + 1) * (
                    self.center[:, 1] + random.randint(-self.center_deviation, self.center_deviation))
        else:
            self.matrix[0, 2] = (-scale_factor_x + 1) * self.center[:, 0]
            self.matrix[1, 2] = (-scale_factor_y + 1) * self.center[:, 1]
        return self.matrix

    def need_data_info(self) -> bool:
        return self.config

    @staticmethod
    def switch_points():
        return None


class RotatePipeline(PipelineOperation):
    """Rotate the input image-mask-keypoints and 2d data by the input degrees"""

    def __init__(self, degrees: int, center: torch.tensor = None, probability: float = 1):
        """
        :param degrees: degrees of the rotation
        :param center : coordinates of the center of scaling. Default: center of the image
        :param probability :[0-1] probability of applying the transform. Default: 1.
        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        self.degrees = degrees
        self.degrees = degrees * one_torch
        self.new_row = torch.zeros(1, 3).to(device)
        self.new_row[:, 2] = 1
        if center is None:
            self.config = True
            self.center = ones_torch.clone()
        else:
            self.config = False
            self.center = center

    def get_op_matrix(self) -> torch.tensor:
        return self.matrix

    def need_data_info(self) -> bool:
        return self.config

    def config_parameters(self, data_info: dict):
        self.center[..., 0] = data_info['shape'][-3] // 2  # x
        self.center[..., 1] = data_info['shape'][-2] // 2
        self.matrix = torch.cat(
            ((kornia.geometry.get_rotation_matrix2d(angle=self.degrees.reshape(1), center=self.center,
                                                    scale=one_torch.reshape(1))).reshape(2, 3),
             self.new_row))
        self.config = False

    @staticmethod
    def switch_points():
        return None


class RandomRotatePipeline(PipelineOperation):
    """Rotate the input image-mask-keypoints and 2d data by a random scaling value calculated within the input range"""

    def __init__(self, degrees_range: tuple, probability: float = 1, center_deviation: int = None,
                 center: torch.tensor = None):
        """

        :param degrees_range: range of degrees to apply
        :param probability: [0-1]  probability of applying the transform. Default: 1.
        :param center_deviation : produces random deviations at the rotating center. The deviations will be a maximum
                of the number of pixels indicated in this parameter
        :param center : coordinates of the center of scaling. Default: center of the image

        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        self.center_deviation = center_deviation
        if not isinstance(degrees_range, tuple):
            raise Exception("Degrees range must be a tuple (min, max)")
        self.degrees_range = degrees_range
        self.new_row = torch.zeros(1, 3).to(device)
        self.new_row[:, 2] = 1
        if center is None:
            self.config = True
            self.center = ones_torch.clone()
        else:
            self.config = False
            self.center = center

    def get_op_matrix(self) -> torch.tensor:
        degrees = random.randint(self.degrees_range[0], self.degrees_range[1]) * one_torch
        center = self.center
        if self.center_deviation is not None:
            center += random.randint(-self.center_deviation, self.center_deviation)
        return torch.cat(((kornia.geometry.get_rotation_matrix2d(angle=degrees.resize_(1), center=center,
                                                                 scale=one_torch.reshape(1))).reshape(2, 3),
                          self.new_row))

    def need_data_info(self) -> bool:
        return self.config

    def config_parameters(self, data_info: dict):
        self.center[..., 0] = data_info['shape'][-3] // 2  # x
        self.center[..., 1] = data_info['shape'][-2] // 2
        self.config = False

    @staticmethod
    def switch_points():
        return None


class TranslatePipeline(PipelineOperation):
    """Translate the input image-mask-keypoints and 2d data by the input translation"""

    def __init__(self, translation: tuple, probability: float = 1):
        """

        :param translation: pixels to be translated ( translation X, translation Y)
        :param probability:[0-1]  probability of applying the transform. Default: 1.
        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        self.matrix = get_translation_matrix(translation)

    def get_op_matrix(self) -> torch.tensor:
        return self.matrix

    @staticmethod
    def need_data_info() -> bool:
        return False

    @staticmethod
    def switch_points():
        return None


class RandomTranslatePipeline(PipelineOperation):
    """Translate the input image-mask-keypoints and 2d data by a random translation value calculated within
    the input range"""

    def __init__(self, probability: float, translation_range: tuple, same_translation_on_axis: bool = False):
        """

        :param probability:[0-1]  probability of applying the transform. Default: 1.
        :param translation_range: range of pixels to be translated ( min translation, max translation).
                                  Translation X and translation Y are calculated within this range
        :param same_translation_on_axis : whether the translation must be equal in both axes
        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        if not isinstance(translation_range, tuple):
            raise Exception("Translation range must be a tuple (min, max)")
        self.translation_range = translation_range
        self.keep_dim = same_translation_on_axis
        self.matrix = identity.clone()

    def get_op_matrix(self):
        translation_x = random.uniform(self.translation_range[0], self.translation_range[1])
        if self.keep_dim:
            translation_y = translation_x
        else:
            translation_y = random.uniform(self.translation_range[0], self.translation_range[1])
        self.matrix[0, 2] = translation_x
        self.matrix[1, 2] = translation_y
        return self.matrix

    @staticmethod
    def need_data_info() -> bool:
        return False

    @staticmethod
    def switch_points():
        return None


class ShearPipeline(PipelineOperation):
    """Shear the input image-mask-keypoints and 2d data by the input shear factor"""

    def __init__(self, shear: tuple, probability: float = 1):
        """

        :param shear : range of pixels to be apply on the shear ( shear X, shear Y).
        :param probability :[0-1]  probability of applying the transform. Default: 1.
        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        self.matrix = get_squared_shear_matrix(shear)

    def get_op_matrix(self) -> torch.tensor:
        return self.matrix

    @staticmethod
    def need_data_info() -> bool:
        return False

    @staticmethod
    def switch_points():
        return None


class RandomShearPipeline(PipelineOperation):
    """Shear the input image-mask-keypoints and 2d data by a random shear value calculated within the input range"""

    def __init__(self, probability: float, shear_range: tuple):
        """

        :param probability: [0-1] probability of applying the transform. Default: 1.
        :param shear_range : range of pixels to be apply on the shear ( shear X, shear Y).
        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        if not isinstance(shear_range, tuple):
            raise Exception("Translation range must be a tuple (min, max)")
        self.shear_range = shear_range
        self.matrix = identity.clone()

    def get_op_matrix(self) -> torch.tensor:
        self.matrix[0, 1] = random.uniform(self.shear_range[0], self.shear_range[1])  # x
        self.matrix[1, 0] = random.uniform(self.shear_range[0], self.shear_range[1])  # y
        return self.matrix

    @staticmethod
    def need_data_info() -> bool:
        return False

    @staticmethod
    def switch_points():
        return None


class HflipPipeline(PipelineOperation):
    """Horizontally flip the input image-mask-keypoints and 2d data"""

    def __init__(self, probability: float = 1, exchange_points: tuple = None):
        """

        :param probability: [0-1] probability of applying the transform. Default: 1.
        :param exchange_points: if it is not None, it serves as a tuple of the point positions to be exchanged after
                the operation. For example, with the value (1,8) point 1 is exchanged for 8. It is useful for example
                in symmetric images where a point can indicate the left edge. After the flip, that point will be
                the right edge
        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        self.config = True
        self.matrix = identity.clone()
        self.matrix[0, 0] = -1
        self.exchange_points = exchange_points

    def need_data_info(self) -> bool:
        return self.config

    def config_parameters(self, data_info: dict):
        self.matrix[0, 2] = data_info['shape'][0]
        self.config = False

    def get_op_matrix(self) -> torch.tensor:
        return self.matrix

    def switch_points(self):
        return self.exchange_points


class VflipPipeline(PipelineOperation):
    """Vertically flip the input image-mask-keypoints and 2d data"""

    def __init__(self, probability: float, exchange_points: tuple = None):
        """

        :param probability: [0-1] probability of applying the transform. Default: 1.
        :param exchange_points: iif it is not None, it serves as a tuple of the point positions to be exchanged after
                the operation. For example, with the value (1,8) point 1 is exchanged for 8. It is useful for example
                in symmetric images where a point can indicate the left edge. After the flip, that point will be
                the right edge
        """
        PipelineOperation.__init__(self, probability=probability, op_type='geometry')
        self.matrix = identity.clone()
        self.config = True
        self.matrix[1, 1] = -1
        self.exchange_points = exchange_points

    def need_data_info(self) -> bool:
        return self.config

    def config_parameters(self, data_info: dict):
        self.matrix[1, 2] = data_info['shape'][1]
        self.config = False

    def get_op_matrix(self) -> torch.tensor:
        return self.matrix

    def switch_points(self):
        return self.exchange_points
