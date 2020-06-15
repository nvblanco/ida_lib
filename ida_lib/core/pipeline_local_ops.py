from typing import Optional

import numpy as np

from ida_lib.core.pipeline_operations import PipelineOperation
from ida_lib.operations import pixel_ops_functional

__all__ = ['BlurPipeline',
           'GaussianBlurPipeline',
           'GaussianNoisePipeline',
           'PoissonNoisePipeline',
           'SaltAndPepperNoisePipeline',
           'SpekleNoisePipeline']


class GaussianNoisePipeline(PipelineOperation):
    """Add gaussian noise to the input image
    (gaussian noise is a statistical noise having a probability density function (PDF) equal to that of the normal \
     distribution)"""

    def __init__(self, probability: float = 1, var: float = 0.5):
        """

        :param probability:[0-1]   probability of applying the transform. Default: 1.:
        :param var: [0-10 ...]   intensity of noise (0 is no noise)
        """
        PipelineOperation.__init__(self, probability=probability, op_type='independent_op')
        self.var = var

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = pixel_ops_functional.apply_gaussian_noise(img, self.var)
        return img


class SaltAndPepperNoisePipeline(PipelineOperation):
    """Add salt and pepper noise to the input image
    (salt-and-pepper noise is a statistical noise compose of white (salt) and black (pepper) pixels)"""

    def __init__(self, probability=1, amount: Optional[float] = 0.01, s_vs_p: Optional[float] = 0.5):
        """

        :param probability: [0-1] probability of applying the transform. Default: 1.
        :param amount:  [0-1]noise percentage compared to the total number of pixels in the image
               * 0 is no noise
               * 1 is total noise
        :param s_vs_p: [0-1]  percentage of salt (white pixels) res
        """
        PipelineOperation.__init__(self, probability=probability, op_type='independent_op')
        self.amount = amount
        self.s_vs_p = s_vs_p

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = pixel_ops_functional.apply_salt_and_pepper_noise(img, self.amount, self.s_vs_p)
        return img


class SpekleNoisePipeline(PipelineOperation):
    """Add spekle noise to the input image
            (Speckle is a granular interference that inherently exists in and degrades the quality of the active radar,
            synthetic aperture radar (SAR), medical ultrasound and optical coherence tomography images.
            It is applied by adding the image multiplied by the noise matrix -> img + img * uniform_noise)"""

    def __init__(self, probability: float = 1, mean: Optional[float] = 0, var: Optional[float] = 0.01):
        """

        :param probability : [0-1]   : probability of applying the transform. Default: 1.
        :param mean : Mean of random distribution.  default=0
        :param var  : Variance of random distribution. Default: 0.01
        """
        PipelineOperation.__init__(self, probability=probability, op_type='independent_op')
        self.mean = mean
        self.var = var

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = pixel_ops_functional.apply_spekle_noise(img)
        return img


class PoissonNoisePipeline(PipelineOperation):
    """Add poison noise to the input image
        ( It is applied by adding Poisson-distributed noise)"""

    def __init__(self, probability: float = 1):
        """

        :param probability: [0-1] probability of applying the transform. Default: 1.
        """
        PipelineOperation.__init__(self, probability=probability, op_type='independent_op')

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = pixel_ops_functional.apply_poisson_noise(img)
        return img


class GaussianBlurPipeline(PipelineOperation):
    """Blur input image by a Gaussian function"""

    def __init__(self, probability: float = 1, blur_size: tuple = (5, 5)):
        """

        :param probability :[0-1] probability of applying the transform. Default: 1.
        :param blur_size   : size of the square os pixels used to blur each pixel Default: (5,5)
        """
        PipelineOperation.__init__(self, probability=probability, op_type='independent_op')
        self.blur_size = blur_size

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = pixel_ops_functional.apply_gaussian_blur(img, blur_size=self.blur_size)
        return img


class BlurPipeline(PipelineOperation):
    """Blur input image ( non-weighted blur) """

    def __init__(self, probability: float = 1, blur_size: tuple = (5, 5)):
        """

        :param probability: [0-1] probability of applying the transform. Default: 1.
        :param blur_size  : size of the square os pixels used to blur each pixel Default: (5,5)
        """
        PipelineOperation.__init__(self, probability=probability, op_type='independent_op')
        self.blur_size = blur_size

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def apply_to_image_if_probability(self, img: np.ndarray) -> np.ndarray:
        if PipelineOperation.apply_according_to_probability(self):
            img = pixel_ops_functional.apply_blur(img, blur_size=self.blur_size)
        return img
