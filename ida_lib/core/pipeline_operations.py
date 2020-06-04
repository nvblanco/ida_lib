from abc import ABC, abstractmethod
import torch
import kornia
import random
import numpy as np
from ida_lib.operations import utils

__all__ = ['HflipPipeline',
           'VflipPipeline',
           'RotatePipeline',
           'ShearPipeline',
           'ScalePipeline',
           'TranslatePipeline',
           'RandomScalePipeline',
           'RandomRotatePipeline',
           'RandomShearPipeline',
           'RandomTranslatePipeline',
           'ContrastPipeline',
           'RandomContrastPipeline',
           'BrightnessPipeline',
           'RandomBrightnessPipeline',
           'GammaPipeline',
           'RandomGammaPipeline',
           'BlurPipeline',
           'GaussianBlurPipeline',
           'GaussianNoisePipeline',
           'PoissonNoisePipeline',
           'SaltAndPepperNoisePipeline',
           'SpekleNoisePipeline',
           'NormalizePipeline',
           'DesnormalizePipeline']

device = 'cuda'
cuda = torch.device('cuda')
one_torch = torch.tensor(1, device=cuda)
ones_torch = torch.ones(1, 2, device=cuda)
identity = torch.eye(3, 3, device=cuda)
global pixel_value_range
pixel_value_range = (0, 127, 255)  # Default uint8


class PipelineOperation(ABC):
    """Abstract class of pipeline operations"""

    def __init__(self, type: str, probability: float = 1):
        """
        :param type (str) : internal parameter to determine how to treat each operation.
            'geometry' | 'color' | 'Normalize' | 'Independient'
                - Geometry      : operations that can be applied by transformation matrix
                - Color         : operations that can be applied using a pixel-by-pixel mathematical formula
                - Normalize     : normalization operation to be applied last within color operations
                - Independient  : concrete operations with direct implementations
        :param probability (float) : probability of applying the transform. Default: 1.
        """
        self.probability = probability
        self.type = type


    @abstractmethod
    def get_op_matrix(self):
        pass

    def get_op_type(self):
        return self.type

    """ returns a boolean based on a random number that determines whether or not to apply the operation"""

    def apply_according_to_probability(self) -> bool:
        return random.uniform(0, 1) < self.probability


"""
--------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------COLOR OPERATIONS-------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------
"""


class ContrastPipeline(PipelineOperation):
    """Change the contrast of the input image."""

    def __init__(self, contrast_factor: float, probability: float = 1):
        """
        :param probability             : probability of applying the transform. Default: 1.
        :param contrast_factor (float) : modification factor to be applied to the image contrast
            * 0  :total contrast removal
            * 1  :dont modify
            * >1 :aument contrast
        """
        PipelineOperation.__init__(self, probability=probability, type='color')
        self.contrast = contrast_factor

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x: int) -> float: return self.contrast * (x - pixel_value_range[1]) + \
                                                          pixel_value_range[1]


class RandomContrastPipeline(PipelineOperation):
    """Change the contrast of the input image with a random contrast factor calculated within the input range"""

    def __init__(self, contrast_range: tuple, probability: float = 1):
        """
        :param probability (float) [0-1]       : probability of applying the transform. Default: 1.
        :param contrast_range (float tuple)    : range  of modification factor to be applied to the image contrast
                * 0  :total contrast removal
                * 1  :dont modify
                * >1 :aument contrast
        """
        PipelineOperation.__init__(self, probability=probability, type='color')
        if not isinstance(contrast_range, tuple) or len(contrast_range) != 2:
            raise Exception("Contrast factor must be tuple of 2 elements")
        self.contrast_range = contrast_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x):
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
        return contrast * (x - pixel_value_range[1]) + pixel_value_range[1]


class BrightnessPipeline(PipelineOperation):
    """Change brightness of the input image """

    def __init__(self, brightness_factor: float, probability: float = 1):
        """
        :param probability (float) [0-1]       : probability of applying the transform. Default: 1.
        :param brightness_factor (float) [0-2] : desired amount of brightness for the image
                 0 - no brightness
                 1 - same
                 2 - max brightness
        """
        PipelineOperation.__init__(self, probability=probability, type='color')
        self.brigthness = utils.map_value(brightness_factor, 0, 2, -256, 256)

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def get_op_type(self):
        return 'color'

    def transform_function(self, x: int) -> float: return x + self.brigthness


class RandomBrightnessPipeline(PipelineOperation):
    """Change brightness of the input image to random amount calculated within the input range
                Args:
                    probability (float) [0-1]       : probability of applying the transform. Default: 1.
                    brightness_factor (float) [0-2] :
        """

    def __init__(self, probability: float, brightness_range: tuple):
        """
        :param probability (float) [0-1]       : probability of applying the transform. Default: 1.
        :param brightness_range (tuple)        : range of desired amount of brightness for the image
                            0 - no brightness
                            1 - same
                            2 - max brightness
        """
        PipelineOperation.__init__(self, probability=probability, type='color')
        if not isinstance(brightness_range, tuple) or len(brightness_range) != 2:
            raise Exception("Contrast factor must be tuple of 2 elements")
        self.brightness_range = (utils.map_value(brightness_range[0], 0, 2, -256, 256),
                                 utils.map_value(brightness_range[1], 0, 2, -pixel_value_range[2],
                                                 pixel_value_range[2]))

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def get_op_type(self):
        return 'color'

    def transform_function(self, x: int) -> float:
        brigthness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        return x + brigthness


class GammaPipeline(PipelineOperation):
    """Change the luminance of the input image """

    def __init__(self, gamma_factor: float, probability: float = 1):
        """
        :param probability (float) [0-1]       : probability of applying the transform. Default: 1.
        :param gamma_factor (float) (0-5..]    : desired amount of factor gamma for the image
                                0  - no contrast
                                1  - same
                                >1 - more luminance
        """
        PipelineOperation.__init__(self, probability=probability, type='color')
        self.gamma = gamma_factor

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x: int) -> float: return pow(x / pixel_value_range[2], self.gamma) * pixel_value_range[
        2]


class RandomGammaPipeline(PipelineOperation):
    """Change the luminance of the input image by a random gamma factor calculated within the input range"""

    def __init__(self, gamma_range: tuple, probability: float = 1):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param gamma_range (float) (0-5..] : range of desired amount of factor gamma for the image
                            0  - no contrast
                            1  - same
                            >1 - more luminance
        """
        PipelineOperation.__init__(self, probability=probability, type='color')
        if not isinstance(gamma_range, tuple) or len(gamma_range) != 2:
            raise Exception("Contrast factor must be tuple of 2 elements")
        self.gamma_range = gamma_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x: int) -> float:
        gamma = random.randint(self.gamma_range[0], self.gamma_range[1])
        return pow(x / pixel_value_range[2], gamma) * pixel_value_range[2]


class NormalizePipeline(PipelineOperation):
    """Change the pixels value to a normalize range"""

    def __init__(self, probability: float = 1, old_range: tuple = (0, pixel_value_range[2]), new_range: tuple = (0, 1)):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param old_range (int tuple)       : actual range of pixels of the input image. Default: 0-255
        :param new_range (int tuple)       : desired range of pixels of the input image. Default: 0-1
        """
        PipelineOperation.__init__(self, probability=probability, type='normalize')
        self.new_range = new_range
        self.old_range = old_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x: int) -> float: return (x + self.old_range[0]) / (
                self.old_range[1] - self.old_range[0])


class DesnormalizePipeline(PipelineOperation):
    """Desnormalize pixel value"""

    def __init__(self, probability: float = 1, old_range: tuple = (0, 1), new_range: tuple = (0, pixel_value_range[2])):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param old_range (int tuple)       : actual range of pixels of the input image. Default: 0-1
        :param new_range (int tuple)       : desired range of pixels of the input image. Default: 0-255
        """
        PipelineOperation.__init__(self, probability=probability, type='normalize')
        self.new_range = new_range
        self.old_range = old_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x: int) -> float: return (x + self.old_range[0]) / (
                self.old_range[1] - self.old_range[0])


"""
--------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------INDEPENDENT OPERATIONS-----------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------
"""


class GaussianNoisePipeline(PipelineOperation):
    """Add gaussian noise to the input image
    (gaussian noise is a statistical noise having a probability density function (PDF) equal to that of the normal distribution)"""

    def __init__(self, probability: float = 1, var: float = 0.5):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.:
        :param var (float) [0-10 ...]      : intensity of noise (0 is no noise)
        """
        PipelineOperation.__init__(self, probability=probability, type='independent_op')
        self.var = var

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = utils._apply_gaussian_noise(img, self.var)
        return img


class SaltAndPepperNoisePipeline(PipelineOperation):
    """Add salt and pepper noise to the input image
    (salt-and-pepper noise is a statistical noise compose of white (salt) and black (pepper) pixels)"""

    def __init__(self, probability=1, amount=0.01, s_vs_p=0.5):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param amount (float) [0-1]        : noise percentage compared to the total number of pixels in the image
               * 0 is no noisse
               * 1 is total noise
        :param s_vs_p (float) [0-1]         : percentage of salt (white pixels) res
        """
        PipelineOperation.__init__(self, probability=probability, type='independent_op')
        self.amount = amount
        self.s_vs_p = s_vs_p

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = utils._apply_salt_and_pepper_noise(img, self.amount, self.s_vs_p)
        return img


class SpekleNoisePipeline(PipelineOperation):
    """Add spekle noise to the input image
            (Speckle is a granular interference that inherently exists in and degrades the quality of the active radar,
            synthetic aperture radar (SAR), medical ultrasound and optical coherence tomography images.
            It is applied by adding the image multiplied by the noise matrix -> img + img * uniform_noise)"""

    def __init__(self, probability: float = 1, mean: float = 0, var: float = 0.01):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param mean (float, optional)      : Mean of random distribution.  default=0
        :param var (float, optional)       : Variance of random distribution. Default: 0.01
        """
        PipelineOperation.__init__(self, probability=probability, type='independent_op')
        self.mean = mean
        self.var = var

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = utils._apply_spekle_noise(img)
        return img


class PoissonNoisePipeline(PipelineOperation):
    """Add poison noise to the input image
        (Speckle is a granular interference that inherently exists in and degrades the quality of the active radar,
        synthetic aperture radar (SAR), medical ultrasound and optical coherence tomography images.
        It is applied by adding Poisson-distributed noise)"""

    def __init__(self, probability: float = 1):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        """
        PipelineOperation.__init__(self, probability=probability, type='independent_op')

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = utils._apply_poisson_noise(img)
        return img


class GaussianBlurPipeline(PipelineOperation):
    """Blur input image by a Gaussian function"""

    def __init__(self, probability: float = 1, blur_size: tuple = (5, 5)):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param blur_size  (int tuple)      : size of the square os pixels used to blur each pixel Default: (5,5)
        """
        PipelineOperation.__init__(self, probability=probability, type='independent_op')
        self.blur_size = blur_size

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = utils.apply_gaussian_blur(img, blur_size=self.blur_size)
        return img


class BlurPipeline(PipelineOperation):
    """Blur input image ( non-weighted blur) """

    def __init__(self, probability: float = 1, blur_size: tuple = (5, 5)):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param blur_size  (int tuple)      : size of the square os pixels used to blur each pixel Default: (5,5)
        """
        PipelineOperation.__init__(self, probability=probability, type='independent_op')
        self.blur_size = blur_size

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = utils._apply_blur(img, blur_size=self.blur_size)
        return img


"""
---------------------------------------------------------------------------------------------------------------------
----------------------------------------------GEOMETRIC OPERATIONS---------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
"""


class ScalePipeline(PipelineOperation):
    """Scale the input image-mask-keypoints and 2d data by the input scaling value"""

    def __init__(self, scale_factor: float, probability: float = 1, center: torch.tensor = None):
        """
        :param scale_factor (float)        : scale value
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param center (torch tensor)       : coordinates of the center of scaling. Default: center of the image
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
        if center is None:
            self.config = True
            self.center = ones_torch
        else:
            self.center = center
            self.config = False
        self.matrix = identity.clone()
        self.ones_2 = torch.ones(2, device=cuda)
        self.scale_factor = self.ones_2
        if isinstance(scale_factor,
                      float) or isinstance(scale_factor,
                      int):  # si solo se proporciona un valor; se escala por igual en ambos ejes
            self.scale_factor = self.ones_2 * scale_factor
        else:
            self.scale_factor[0] = one_torch * scale_factor[0]
            self.scale_factor[1] = one_torch * scale_factor[1]

    def config_parameters(self, data_info: dict):
        self.config = False
        self.center[..., 1] = data_info['shape'][-1] // 2
        self.center[..., 0] = data_info['shape'][-2] // 2

    def get_op_matrix(self):
        self.matrix[0, 0] = self.scale_factor[0]
        self.matrix[1, 1] = self.scale_factor[1]
        self.matrix[0, 2] = (-self.scale_factor[0] + 1) * self.center[:, 0]
        self.matrix[1, 2] = (-self.scale_factor[1] + 1) * self.center[:, 1]
        return self.matrix

    def need_data_info(self):
        return self.config


class RandomScalePipeline(PipelineOperation):
    """Scale the input image-mask-keypoints and 2d data by a random scaling value calculated within the input range"""

    def __init__(self, probability: float, scale_range: tuple, keep_aspect: bool = True, center_desviation: int = None,
                 center:torch.tensor  = None):
        """
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param scale_factor (float)        : scale value
        :param keep_aspect (boolean)       : whether the scaling should be the same on the X axis and on the Y axis. Default: true
        :param center desviation (int)     : produces random deviations at the scaling center. The deviations will be a maximum of the number of pixels indicated in this parameter
        :param center (torch tensor)       : coordinates of the center of scaling. Default: center of the image
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
        self.keep_aspect = keep_aspect
        if center is None:
            self.config = True
            self.center = ones_torch
        else:
            self.center = center
            self.config = False
        self.matrix = identity.clone()
        self.ones_2 = torch.ones(2, device=cuda)
        self.scale_factor = self.ones_2
        if not isinstance(scale_range, tuple):
            raise Exception("Scale range must be a tuple")
        else:
            self.scale_factor[0] = one_torch * scale_range[0]
            self.scale_factor[1] = one_torch * scale_range[1]
        self.center_desviation = center_desviation

    def config_parameters(self, data_info: dict):
        self.config = False
        self.center[..., 1] = data_info['shape'][-1] // 2
        self.center[..., 0] = data_info['shape'][-2] // 2

    def get_op_matrix(self) -> torch.tensor:
        scale_factor_x = random.uniform(self.scale_factor[0], self.scale_factor[1])
        if self.keep_aspect:
            scale_factor_y = scale_factor_x
        else:
            scale_factor_y = random.uniform(self.scale_factor[0], self.scale_factor[1])
        self.matrix[0, 0] = scale_factor_x
        self.matrix[1, 1] = scale_factor_y
        if self.center_desviation is not None:
            self.center[:, 0] += random.randint(0, self.center_desviation)
            self.matrix[0, 2] = (-scale_factor_x + 1) * (
                    self.center[:, 0] + random.randint(-self.center_desviation, self.center_desviation))
            self.matrix[1, 2] = (-scale_factor_y + 1) * (
                    self.center[:, 1] + random.randint(-self.center_desviation, self.center_desviation))
        else:
            self.matrix[0, 2] = (-scale_factor_x + 1) * self.center[:, 0]
            self.matrix[1, 2] = (-scale_factor_y + 1) * self.center[:, 1]
        return self.matrix

    def need_data_info(self) -> bool:
        return self.config


class RotatePipeline(PipelineOperation):
    """Rotate the input image-mask-keypoints and 2d data by the input degrees"""

    def __init__(self, degrees: int, center: torch.tensor  = None, probability: float = 1):
        """
        :param degrees (float)             : degrees of the rotation
        :param center (torch tensor)       : coordinates of the center of scaling. Default: center of the image
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
        self.degrees = degrees
        self.degrees = degrees * one_torch
        self.new_row = torch.Tensor(1, 3).to(device)
        if center is None:
            self.config = True
            self.center = ones_torch.clone()
        else:
            self.config = False
            self.center = center

    def get_op_matrix(self) -> torch.tensor:
        return torch.cat(((kornia.geometry.get_rotation_matrix2d(angle=self.degrees.reshape(1), center=self.center,
                                                                 scale=one_torch.reshape(1))).reshape(2, 3), self.new_row))

    def need_data_info(self) -> bool:
        return self.config

    def config_parameters(self, data_info: dict):
        self.center[..., 0] = data_info['shape'][-2] // 2  # x
        self.center[..., 1] = data_info['shape'][-1] // 2
        self.config = False


class RandomRotatePipeline(PipelineOperation):
    """Rotate the input image-mask-keypoints and 2d data by a random scaling value calculated within the input range"""

    def __init__(self, degrees_range: tuple, probability: float=1, center_desviation: int =None, center: torch.tensor =None):
        """

        :param degrees_range (float)       : range of degrees to apply
        :param probability (float) [0-1]   : probability of applying the transform. Default: 1.
        :param center desviation (int)     : produces random deviations at the rotating center. The deviations will be a maximum of the number of pixels indicated in this parameter
        :param center (torch tensor)       : coordinates of the center of scaling. Default: center of the image
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
        self.center_desviation = center_desviation
        if not isinstance(degrees_range, tuple):
            raise Exception("Degrees range must be a tuple (min, max)")
        self.degrees_range = degrees_range
        self.new_row = torch.Tensor(1, 3).to(device)
        if center is None:
            self.config = True
            self.center = ones_torch.clone()
        else:
            self.config = False
            self.center = center

    def get_op_matrix(self) -> torch.tensor:
        degrees = random.randint(self.degrees_range[0], self.degrees_range[1]) * one_torch
        center = self.center
        if self.center_desviation is not None:
            center += random.randint(-self.center_desviation, self.center_desviation)
        return torch.cat(((kornia.geometry.get_rotation_matrix2d(angle=degrees.resize_(1), center=center,
                                                                 scale=one_torch)).reshape(2, 3), self.new_row))

    def need_data_info(self) -> bool:
        return self.config

    def config_parameters(self, data_info: dict):
        self.center[..., 0] = data_info['shape'][-2] // 2  # x
        self.center[..., 1] = data_info['shape'][-1] // 2
        self.config = False


class TranslatePipeline(PipelineOperation):
    """Translate the input image-mask-keypoints and 2d data by the input translation"""

    def __init__(self, translation: tuple, probability: float = 1):
        """

        :param translation (tuple float) : pixels to be translated ( translation X, translation Y)
        :param probability (float) [0-1] : probability of applying the transform. Default: 1.
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
        self.translation_x = translation[0] * one_torch
        self.translation_y = translation[1] * one_torch
        self.matrix = identity.clone()

    def get_op_matrix(self) -> torch.tensor:
        self.matrix[0, 2] = self.translation_x
        self.matrix[1, 2] = self.translation_y
        return self.matrix

    def need_data_info(self) -> bool:
        return False


class RandomTranslatePipeline(PipelineOperation):
    """Translate the input image-mask-keypoints and 2d data by a random translation value calculated within the input range"""

    def __init__(self, probability: float, translation_range: tuple, same_translation_on_axis: bool = False):
        """

        :param probability (float) [0-1]            : probability of applying the transform. Default: 1.
        :param translation_range(tuple float)       : range of pixels to be translated ( min translation, max translation). Translation X and translation Y are calculated within this range
        :param same_translation_on_axis (boolean)   : whether the translation must be equal in both axes
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
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

    def need_data_info(self) -> bool:
        return False


class ShearPipeline(PipelineOperation):
    """Shear the input image-mask-keypoints and 2d data by the input shear factor"""

    def __init__(self, shear: tuple, probability: float = 1):
        """

        :param shear (tuple float)                  : range of pixels to be apply on the shear ( shear X, shear Y).
        :param probability (float) [0-1]            : probability of applying the transform. Default: 1.
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
        self.shear_x = shear[0]
        self.shear_y = shear[1]
        self.matrix = identity.clone()

    def get_op_matrix(self) -> torch.tensor:
        self.matrix[0, 1] = self.shear_x
        self.matrix[1, 0] = self.shear_y
        return self.matrix

    def need_data_info(self) -> bool:
        return False


class RandomShearPipeline(PipelineOperation):
    """Shear the input image-mask-keypoints and 2d data by a random shear value calculated within the input range"""

    def __init__(self, probability: float, shear_range: tuple):
        """

        :param probability (float) [0-1]      : probability of applying the transform. Default: 1.
        :param shear (tuple float)            : range of pixels to be apply on the shear ( shear X, shear Y).
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
        if not isinstance(shear_range, tuple):
            raise Exception("Translation range must be a tuple (min, max)")
        self.shear_range = shear_range
        self.matrix = identity.clone()

    def get_op_matrix(self) -> torch.tensor:
        self.matrix[0, 1] = random.uniform(self.shear_range[0], self.shear_range[1])  # x
        self.matrix[1, 0] = random.uniform(self.shear_range[0], self.shear_range[1])  # y
        return self.matrix

    def need_data_info(self) -> bool:
        return False


class HflipPipeline(PipelineOperation):
    """Horizontally flip the input image-mask-keypoints and 2d data"""

    def __init__(self, probability: float = 1):
        """

        :param probability (float) [0-1]      : probability of applying the transform. Default: 1.
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
        self.config = True
        self.matrix = identity.clone()
        self.matrix[0, 0] = -1

    def need_data_info(self) -> bool:
        return self.config

    def config_parameters(self, data_info: dict):
        self.matrix[0, 2] = data_info['shape'][1]
        self.config = False

    def get_op_matrix(self) -> torch.tensor:
        return self.matrix


class VflipPipeline(PipelineOperation):
    """Vertically flip the input image-mask-keypoints and 2d data"""

    def __init__(self, probability: float):
        """

        :param probability (float) [0-1]      : probability of applying the transform. Default: 1.
        :param heigth:
        """
        PipelineOperation.__init__(self, probability=probability, type='geometry')
        self.matrix = identity.clone()
        self.config = True
        self.matrix[1, 1] = -1

    def need_data_info(self) -> bool:
        return self.config

    def config_parameters(self, data_info: dict):
        self.matrix[1, 2] = data_info['shape'][0]
        self.config = False

    def get_op_matrix(self) -> torch.tensor:
        return self.matrix