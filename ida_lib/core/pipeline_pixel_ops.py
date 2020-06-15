import random

from ida_lib.core.pipeline_operations import PipelineOperation
from ida_lib.operations import utils

__all__ = ['ContrastPipeline',
           'RandomContrastPipeline',
           'BrightnessPipeline',
           'RandomBrightnessPipeline',
           'GammaPipeline',
           'RandomGammaPipeline',
           'NormalizePipeline',
           'DenormalizePipeline']


class ContrastPipeline(PipelineOperation):
    """Change the contrast of the input image."""

    def __init__(self, contrast_factor: float, probability: float = 1):
        """

        :param probability: probability of applying the transform. Default: 1.
        :param contrast_factor: modification factor to be applied to the image contrast
            * 0  :total contrast removal
            * 1  :dont modify
            * >1 :augment contrast
        """
        PipelineOperation.__init__(self, probability=probability, op_type='color')
        self.contrast = contrast_factor

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x: int) -> float: return self.contrast * (x - 127) + 127


class RandomContrastPipeline(PipelineOperation):
    """Change the contrast of the input image with a random contrast factor calculated within the input range"""

    def __init__(self, contrast_range: tuple, probability: float = 1):
        """

        :param probability: probability of applying the transform. Default: 1.
        :param contrast_range: range  of modification factor to be applied to the image contrast
                * 0  :total contrast removal
                * 1  :dont modify
                * >1 :augment contrast
        """
        PipelineOperation.__init__(self, probability=probability, op_type='color')
        if not isinstance(contrast_range, tuple) or len(contrast_range) != 2:
            raise Exception("Contrast factor must be tuple of 2 elements")
        self.contrast_range = contrast_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x):
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
        return contrast * (x - 127) + 127


class BrightnessPipeline(PipelineOperation):
    """Change brightness of the input image """

    def __init__(self, brightness_factor: float, probability: float = 1):
        """

        :param probability: probability of applying the transform. Default: 1.
        :param brightness_factor: desired amount of brightness for the image
                 0 - no brightness
                 1 - same
                 2 - max brightness
        """
        PipelineOperation.__init__(self, probability=probability, op_type='color')
        self.brightness = utils.map_value(brightness_factor, 0, 2, -256, 256)

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def get_op_type(self):
        return 'color'

    def transform_function(self, x: int) -> float: return x + self.brightness


class RandomBrightnessPipeline(PipelineOperation):
    """Change brightness of the input image to random amount calculated within the input range """

    def __init__(self, probability: float, brightness_range: tuple):
        """

        :param probability:[0-1] probability of applying the transform. Default: 1.
        :param brightness_range: range of desired amount of brightness for the image
                            0 - no brightness
                            1 - same
                            2 - max brightness
        """
        PipelineOperation.__init__(self, probability=probability, op_type='color')
        if not isinstance(brightness_range, tuple) or len(brightness_range) != 2:
            raise Exception("Contrast factor must be tuple of 2 elements")
        self.brightness_range = (utils.map_value(brightness_range[0], 0, 2, -256, 256),
                                 utils.map_value(brightness_range[1], 0, 2, -255,
                                                 255))

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def get_op_type(self):
        return 'color'

    def transform_function(self, x: int) -> float:
        brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        return x + brightness


class GammaPipeline(PipelineOperation):
    """Change the luminance of the input image """

    def __init__(self, gamma_factor: float, probability: float = 1):
        """

        :param probability:[0-1] probability of applying the transform. Default: 1.
        :param gamma_factor:[ 0-5..] desired amount of factor gamma for the image
                                0  - no contrast
                                1  - same
                                >1 - more luminance
        """
        PipelineOperation.__init__(self, probability=probability, op_type='color')
        self.gamma = gamma_factor

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x: int) -> float: return pow(x / 255, self.gamma) * 255


class RandomGammaPipeline(PipelineOperation):
    """Change the luminance of the input image by a random gamma factor calculated within the input range"""

    def __init__(self, gamma_range: tuple, probability: float = 1):
        """

        :param probability: [0-1]  probability of applying the transform. Default: 1.
        :param gamma_range:(0-5..] range of desired amount of factor gamma for the image
                            0  - no contrast
                            1  - same
                            >1 - more luminance
        """
        PipelineOperation.__init__(self, probability=probability, op_type='color')
        if not isinstance(gamma_range, tuple) or len(gamma_range) != 2:
            raise Exception("Contrast factor must be tuple of 2 elements")
        self.gamma_range = gamma_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x: int) -> float:
        gamma = random.randint(self.gamma_range[0], self.gamma_range[1])
        return pow(x / 255, gamma) * 255


class NormalizePipeline(PipelineOperation):
    """Change the pixels value to a normalize range"""

    def __init__(self, probability: float = 1, old_range: tuple = (0, 255), new_range: tuple = (0, 1)):
        """

        :param probability: [0-1] probability of applying the transform. Default: 1.
        :param old_range: actual range of pixels of the input image. Default: 0-255
        :param new_range: desired range of pixels of the input image. Default: 0-1
        """
        PipelineOperation.__init__(self, probability=probability, op_type='normalize')
        self.new_range = new_range
        self.old_range = old_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    @staticmethod
    def transform_function(x: int) -> float: return x / 255

    '''def transform_function(self, x: int) -> float: return (x + self.old_range[0]) / (
                self.old_range[1] - self.old_range[0])'''


class DenormalizePipeline(PipelineOperation):
    """Denormalize pixel value"""

    def __init__(self, probability: float = 1, old_range: tuple = (0, 1), new_range: tuple = (0, 255)):
        """

        :param probability: [0-1]  probability of applying the transform. Default: 1.
        :param old_range: actual range of pixels of the input image. Default: 0-1
        :param new_range: desired range of pixels of the input image. Default: 0-255
        """
        PipelineOperation.__init__(self, probability=probability, op_type='normalize')
        self.new_range = new_range
        self.old_range = old_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x: int) -> float: return (x + self.old_range[0]) / (
            self.old_range[1] - self.old_range[0])
