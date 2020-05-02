from abc import ABC, abstractmethod
import torch
import kornia
import random
from operations import utils

device = 'cuda'
cuda = torch.device('cuda')
one_torch = torch.tensor(1, device=cuda)
one_torch = torch.ones(1, device=cuda)
ones_torch = torch.ones(1, 2, device=cuda)
data_types_2d = {"image", "mask", "heatmap"}
identity = torch.eye(3, 3, device=cuda)


class pipeline_operation(ABC):
    def __init__(self, type, probability=1):
        self.probability = probability
        self.type = type

    @abstractmethod
    def get_op_matrix(self):
        pass

    def get_op_type(self):
        return self.type

    def apply_according_to_probability(self):
        return random.uniform(0, 1) < self.probability


class contrast_pipeline(pipeline_operation):
    '''Change the contrast of the input image.

    Args:
        probability (float) [0-1]   : probability of applying the transform. Default: 1.
        contrast_factor (float)     : modification factor to be applied to the image contrast
            * 0  :total contrast removal
            * 1  :dont modify
            * >1 :aument contrast

    Target:
        image
    '''
    def __init__(self, contrast_factor,  probability=1):
        pipeline_operation.__init__(self, probability=probability, type='color')
        self.contrast = contrast_factor

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x):
        contrast = random.randint(self.contrast_range[0], self.contrast_range[1])
        return contrast * (x - 127) + 127

    def transform_function(self, x): return self.contrast * (x - 127) + 127


class random_contrast_pipeline(pipeline_operation):
    '''Change the contrast of the input image with a random contrast factor calculated within the input range

        Args:
            probability (float) [0-1]       : probability of applying the transform. Default: 1.
            contrast_range (float tuple)    : range  of modification factor to be applied to the image contrast
                * 0  :total contrast removal
                * 1  :dont modify
                * >1 :aument contrast

        Target:
            image
        '''
    def __init__(self, contrast_range, probability=1):
        pipeline_operation.__init__(self, probability=probability, type='color')
        if not isinstance(contrast_range, tuple) or len(contrast_range) != 2:
            raise Exception("Contrast factor must be tuple of 2 elements")
        self.contrast_range = contrast_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x):
        contrast = random.randint(self.contrast_range[0], self.contrast_range[1])
        return contrast * (x - 127) + 127


class gaussian_noise_pipeline(pipeline_operation):
    '''Add gaussian noise to the input image
    (gaussian noise is a statistical noise having a probability density function (PDF) equal to that of the normal distribution)

            Args:
                probability (float) [0-1]   : probability of applying the transform. Default: 1.
                var (float) [0-10 ...]      : intensity of noise (0 is no noise)

            Target:
                image
     '''
    def __init__(self, probability=1, var=0.5):
        pipeline_operation.__init__(self, probability=probability, type='independent_op')
        self.var = var

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img):
        if pipeline_operation.apply_according_to_probability(self):
            img = utils._apply_gaussian_noise(img, self.var)
        return img


class salt_and_pepper_noise_pipeline(pipeline_operation):
    '''Add salt and pepper noise to the input image
        (salt-and-pepper noise is a statistical noise compose of white (salt) and black (pepper) pixels)

                Args:
                    probability (float) [0-1]   : probability of applying the transform. Default: 1.
                    amount (float) [0-1]        : noise percentage compared to the total number of pixels in the image
                        0 is no noisse
                        1 is total noise

                Target:
                    image
         '''
    def __init__(self, probability=1, amount=0.01, s_vs_p=0.5):
        pipeline_operation.__init__(self, probability=probability, type='independent_op')
        self.amount = amount
        self.s_vs_p = s_vs_p

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img):
        if pipeline_operation.apply_according_to_probability(self):
            img = utils._apply_salt_and_pepper_noise(img, self.amount, self.s_vs_p)
        return img


class spekle_noise_pipeline(pipeline_operation):
    '''Add spekle noise to the input image
            (Speckle is a granular interference that inherently exists in and degrades the quality of the active radar,
            synthetic aperture radar (SAR), medical ultrasound and optical coherence tomography images.
            It is applied by adding the image multiplied by the noise matrix -> img + img * uniform_noise)

                    Args:
                        probability (float) [0-1]   : probability of applying the transform. Default: 1.
                        mean (float, optional)      : Mean of random distribution.  default=0
                        var (float, optional)       : Variance of random distribution. Default: 0.01
                    Target:
                        image
             '''
    def __init__(self, probability=1, mean=0, var=0.01):
        pipeline_operation.__init__(self, probability=probability, type='independent_op')
        self.mean = mean
        self.var = var

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img):
        if pipeline_operation.apply_according_to_probability(self):
            img = utils._apply_spekle_noise(img)
        return img


class poisson_noise_pipeline(pipeline_operation):
    '''Add poison noise to the input image
                (Speckle is a granular interference that inherently exists in and degrades the quality of the active radar,
                synthetic aperture radar (SAR), medical ultrasound and optical coherence tomography images.
                It is applied by adding Poisson-distributed noise)

                        Args:
                            probability (float) [0-1]   : probability of applying the transform. Default: 1.
                            mean (float, optional)      : Mean of random distribution.  default=0
                            var (float, optional)       : Variance of random distribution. Default: 0.01
                        Target:
                            image
                 '''
    def __init__(self, probability=1):
        pipeline_operation.__init__(self, probability=probability, type='independent_op')

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img):
        if pipeline_operation.apply_according_to_probability(self):
            img = utils._apply_poisson_noise(img)
        return img


class gaussian_blur_pipeline(pipeline_operation):
    '''Blur input image by a Gaussian function
                Args:
                    probability (float) [0-1]   : probability of applying the transform. Default: 1.
                    blur_size  (int tuple)      : size of the square os pixels used to blur each pixel Default: (5,5)
                Target:
                    image
    '''
    def __init__(self, probability=1,  blur_size = (5,5)):
        pipeline_operation.__init__(self, probability=probability, type='independent_op')
        self.blur_size = blur_size

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img):
        if pipeline_operation.apply_according_to_probability(self):
            img = utils.apply_gaussian_blur(img, blur_size=self.blur_size)
        return img


class blur_pipeline(pipeline_operation):
    '''Blur input image ( non-weighted blur)
                    Args:
                        probability (float) [0-1]   : probability of applying the transform. Default: 1.
                        blur_size  (int tuple)      : size of the square os pixels used to blur each pixel Default: (5,5)
                    Target:
                        image
        '''
    def __init__(self, probability=1,  blur_size = (5,5)):
        pipeline_operation.__init__(self, probability=probability, type='independent_op')
        self.blur_size = blur_size

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img):
        if pipeline_operation.apply_according_to_probability(self):
            img = utils._apply_blur(img, blur_size=self.blur_size)
        return img

class brightness_pipeline(pipeline_operation):
    '''Change brightness of the input image
            Args:
                probability (float) [0-1]       : probability of applying the transform. Default: 1.
                brightness_factor (float) [0-2] : desired amount of brightness for the image
                        0 - no brightness
                        1 - same
                        2 - max brightness
    '''
    def __init__(self,  brightness_factor, probability=1,):
        pipeline_operation.__init__(self, probability=probability, type='color')
        self.brigthness = utils.map_value(brightness_factor, 0,2,-256,256)

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def get_op_type(self):
        return 'color'

    def transform_function(self, x): return x + self.brigthness


class random_brightness_pipeline(pipeline_operation):
    '''Change brightness of the input image to random amount calculated within the input range
                Args:
                    probability (float) [0-1]       : probability of applying the transform. Default: 1.
                    brightness_factor (float) [0-2] : desired amount of brightness for the image
                            0 - no brightness
                            1 - same
                            2 - max brightness
        '''
    def __init__(self, probability, brightness_range):
        pipeline_operation.__init__(self, probability=probability, type='color')
        if not isinstance(brightness_range, tuple) or len(brightness_range) != 2:
            raise Exception("Contrast factor must be tuple of 2 elements")
        self.brightness_range = (utils.map_value(brightness_range[0] , 0,2,-256,256), utils.map_value(brightness_range[1], 0, 2, -256, 256))

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def get_op_type(self):
        return 'color'

    def transform_function(self, x):
        brigthness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        return x + brigthness


class gamma_pipeline(pipeline_operation):
    '''Change the luminance of the input image
                    Args:
                        probability (float) [0-1]       : probability of applying the transform. Default: 1.
                        gamma_factor (float) (0-5..]    : desired amount of factor gamma for the image
                                0  - no contrast
                                1  - same
                                >1 - more luminance
            '''
    def __init__(self, gamma_factor, probability=1):
        pipeline_operation.__init__(self, probability=probability, type='color')
        self.gamma = gamma_factor

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x): return pow(x / 255.0, self.gamma) * 255.0


class random_gamma_pipeline(pipeline_operation):
    '''Change the luminance of the input image by a random gamma factor calculated within the input range
                        Args:
                            probability (float) [0-1]   : probability of applying the transform. Default: 1.
                            gamma_range (float) (0-5..] : range of desired amount of factor gamma for the image
                                    0  - no contrast
                                    1  - same
                                    >1 - more luminance
                '''
    def __init__(self, gamma_range, probability=1):
        pipeline_operation.__init__(self, probability=probability, type='color')
        if not isinstance(gamma_range, tuple) or len(gamma_range) != 2:
            raise Exception("Contrast factor must be tuple of 2 elements")
        self.gamma_range = gamma_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x):
        gamma = random.randint(self.gamma_range[0], self.gamma_range[1])
        return pow(x / 255.0, gamma) * 255.0


class normalize_pipeline(pipeline_operation):
    '''Change the pixels value to a normalize range
                        Args:
                            probability (float) [0-1]   : probability of applying the transform. Default: 1.
                            old_range (int tuple)       : actual range of pixels of the input image. Default: 0-255
                            new_range (int tuple)       : desired range of pixels of the input image. Default: 0-1
                    '''
    def __init__(self, probability=1, old_range=(0, 255), new_range=(0, 1)):
        pipeline_operation.__init__(self, probability=probability, type='normalize')
        self.new_range = new_range
        self.old_range = old_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x): return (x + self.old_range[0]) / (self.old_range[1] - self.old_range[0])

class desnormalize_pipeline(pipeline_operation):
    '''Desnormalize pixel values
                    Args:
                            probability (float) [0-1]   : probability of applying the transform. Default: 1.
                            old_range (int tuple)       : actual range of pixels of the input image. Default: 0-1
                            new_range (int tuple)       : desired range of pixels of the input image. Default: 0-255
                        '''
    def __init__(self, probability=1, old_range=(0, 1), new_range=(0, 255)):
        pipeline_operation.__init__(self, probability=probability, type='normalize')
        self.new_range = new_range
        self.old_range = old_range

    def get_op_matrix(self):
        raise Exception("Color operations doesnt have matrix")

    def transform_function(self, x): return (x + self.old_range[0]) / (self.old_range[1] - self.old_range[0])

class resize_pipeline(pipeline_operation):
    def __init__(self, probability, new_size):
        pipeline_operation.__init__(self, probability=probability, type='independent_op')
        self.new_size = new_size

    def get_op_matrix(self):
        raise Exception("Independent operations doesnt have matrix")

    def _apply_to_image_if_probability(self, img):
        if pipeline_operation.apply_according_to_probability(self):
            img = utils._resize_image(img, self.new_size)
        return img


class scale_pipeline(pipeline_operation):
    '''Scale the input image-mask-keypoints and 2d data by the input scaling value

        Args:
            probability (float) [0-1]   : probability of applying the transform. Default: 1.
            scale_factor (float)        : scale value
            center (torch tensor)       : coordinates of the center of scaling. Default: center of the image

    '''
    def __init__(self,  scale_factor, probability=1, center=None):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        if center is None:
            self.config = True
            self.center = ones_torch
        else:
            self.center = center
            self.config = False
        self.matrix = identity.clone()
        self.ones_2 = torch.ones(2, device=cuda)
        self.scale_factor = self.ones_2
        if isinstance(self.scale_factor,
                      float):  # si solo se proporciona un valor; se escala por igual en ambos ejes
            self.scale_factor = self.ones_2 * scale_factor
        else:
            self.scale_factor[0] *= scale_factor[0]
            self.scale_factor[1] *= scale_factor[1]

    def config_parameters(self, data_info):
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


class random_scale_pipeline(pipeline_operation):
    '''Scale the input image-mask-keypoints and 2d data by a random scaling value calculated within the input range

            Args:
                probability (float) [0-1]   : probability of applying the transform. Default: 1.
                scale_factor (float)        : scale value
                center (torch tensor)       : coordinates of the center of scaling. Default: center of the image
                center desviation (int)     : produces random deviations at the scaling center. The deviations will be a maximum of the number of pixels indicated in this parameter
                keep_aspect (boolean)       : whether the scaling should be the same on the X axis and on the Y axis. Default: true
        '''
    def __init__(self, probability, scale_range, keep_aspect=True, center_desviation=None, center=None):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
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

    def config_parameters(self, data_info):
        self.config = False
        self.center[..., 1] = data_info['shape'][-1] // 2
        self.center[..., 0] = data_info['shape'][-2] // 2

    def get_op_matrix(self):
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

    def need_data_info(self):
        return self.config


class rotate_pipeline(pipeline_operation):
    '''Rotate the input image-mask-keypoints and 2d data by the input degrees

           Args:
               probability (float) [0-1]   : probability of applying the transform. Default: 1.
               degrees (float)             : degrees of the rotation
               center (torch tensor)       : coordinates of the center of scaling. Default: center of the image
       '''
    def __init__(self, degrees, center=None, probability=1):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.degrees = degrees
        self.degrees = degrees * one_torch
        self.new_row = torch.Tensor(1, 3).to(device)
        if center is None:
            self.config = True
            self.center = ones_torch.clone()
        else:
            self.config = False
            self.center = center

    def get_op_matrix(self):
        return torch.cat(((kornia.geometry.get_rotation_matrix2d(angle=self.degrees, center=self.center,
                                                                 scale=one_torch)).reshape(2, 3), self.new_row))

    def need_data_info(self):
        return self.config

    def config_parameters(self, data_info):
        self.center[..., 0] = data_info['shape'][-2] // 2  # x
        self.center[..., 1] = data_info['shape'][-1] // 2
        self.config = False


class random_rotate_pipeline(pipeline_operation):
    '''Rotate the input image-mask-keypoints and 2d data by a random scaling value calculated within the input range

               Args:
                   probability (float) [0-1]   : probability of applying the transform. Default: 1.
                   degrees_range (float)       : range of degrees to apply
                   center (torch tensor)       : coordinates of the center of scaling. Default: center of the image
                   center desviation (int)     : produces random deviations at the rotating center. The deviations will be a maximum of the number of pixels indicated in this parameter
           '''
    def __init__(self, degrees_range,  probability=1 ,center_desviation=None, center=None):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
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

    def get_op_matrix(self):
        degrees = random.randint(self.degrees_range[0], self.degrees_range[1]) * one_torch
        center = self.center
        if self.center_desviation is not None:
            center += random.randint(-self.center_desviation, self.center_desviation)
        return torch.cat(((kornia.geometry.get_rotation_matrix2d(angle=degrees, center=center,
                                                                 scale=one_torch)).reshape(2, 3), self.new_row))

    def need_data_info(self):
        return self.config

    def config_parameters(self, data_info):
        self.center[..., 0] = data_info['shape'][-2] // 2  # x
        self.center[..., 1] = data_info['shape'][-1] // 2
        self.config = False


class translate_pipeline(pipeline_operation):
    '''Translate the input image-mask-keypoints and 2d data by the input translation

               Args:
                   probability (float) [0-1] : probability of applying the transform. Default: 1.
                   translation (tuple float) : pixels to be translated ( translation X, translation Y)
    '''
    def __init__(self, translation, probability=1):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.translation_x = translation[0] * one_torch
        self.translation_y = translation[1] * one_torch
        self.matrix = identity.clone()

    def get_op_matrix(self):
        self.matrix[0, 2] = self.translation_x
        self.matrix[1, 2] = self.translation_y
        return self.matrix

    def need_data_info(self):
        return False

class random_translate_pipeline(pipeline_operation):
    '''Translate the input image-mask-keypoints and 2d data by a random translation value calculated within the input range

                   Args:
                       probability (float) [0-1]            : probability of applying the transform. Default: 1.
                       translation (tuple float)            : range of pixels to be translated ( min translation, max translation). Translation X and translation Y are calculated within this range
                       same_translation_on_axis (boolean)   : whether the translation must be equal in both axes
        '''
    def __init__(self, probability, translation_range, same_translation_on_axis = False):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
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

    def need_data_info(self):
        return False


class shear_pipeline(pipeline_operation):
    '''Shear the input image-mask-keypoints and 2d data by the input shear factor

                       Args:
                           probability (float) [0-1]            : probability of applying the transform. Default: 1.
                           shear (tuple float)                  : range of pixels to be apply on the shear ( shear X, shear Y).
            '''
    def __init__(self, shear, probability=1):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.shear_x = shear[0]
        self.shear_y = shear[1]
        self.matrix = identity.clone()

    def get_op_matrix(self):
        self.matrix[0, 1] = self.shear_x
        self.matrix[1, 0] = self.shear_y
        return self.matrix

    def need_data_info(self):
        return False

class random_shear_pipeline(pipeline_operation):
    '''Shear the input image-mask-keypoints and 2d data by a random shear value calculated within the input range

             Args:
                  probability (float) [0-1]      : probability of applying the transform. Default: 1.
                  shear (tuple float)            : range of pixels to be apply on the shear ( shear X, shear Y).
                '''
    def __init__(self, probability, shear_range):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        if not isinstance(shear_range, tuple):
            raise Exception("Translation range must be a tuple (min, max)")
        self.shear_range = shear_range
        self.matrix = identity.clone()

    def get_op_matrix(self):
        self.matrix[0, 1] = random.uniform(self.shear_range[0], self.shear_range[1]) #x
        self.matrix[1, 0] = random.uniform(self.shear_range[0], self.shear_range[1]) #y
        return self.matrix

    def need_data_info(self):
        return False


class hflip_pipeline(pipeline_operation):
    '''Horizontally flip the input image-mask-keypoints and 2d data
             Args:
                  probability (float) [0-1]      : probability of applying the transform. Default: 1.
    '''
    def __init__(self, probability=1):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.config = True
        self.matrix = identity.clone()
        self.matrix[0, 0] = -1
        # self.matrix[0, 2] = self.width

    def need_data_info(self):
        return self.config

    def config_parameters(self, data_info):
        self.matrix[0, 2] = data_info['shape'][1]
        self.config = False

    def get_op_matrix(self):
        return self.matrix


class vflip_pipeline(pipeline_operation):
    '''Vertically flip the input image-mask-keypoints and 2d data
                 Args:
                      probability (float) [0-1]      : probability of applying the transform. Default: 1.
        '''
    def __init__(self, probability, heigth=256):
        pipeline_operation.__init__(self, probability=probability, type='geometry')
        self.matrix = identity.clone()
        self.config = True
        self.matrix[1, 1] = -1
        self.matrix[1, 2] = heigth

    def need_data_info(self):
        return self.config

    def config_parameters(self, data_info):
        self.matrix[1, 2] = data_info['shape'][1]
        self.config = False

    def get_op_matrix(self):
        return self.matrix