from core.pipeline_operations import *
from core.pipeline_functional import *


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
    def __init__(self, pipeline_operations, resize = None):
        self.color_ops, self.geom_ops, self.indep_ops = split_operations_by_type(pipeline_operations)
        self.geom_ops.reverse()  # to apply matrix multiplication in the user order
        self.info_data = None
        self.resize = resize

    def apply_geometry_transform_data2d(self, image, matrix):
        return kornia.geometry.affine(image, matrix[:2, :])

    def apply_geometry_transform_points(self, points_matrix, matrix):
        return torch.matmul(matrix, points_matrix)

    '''
    Applies the transformations to the input image batch. 
        *   If it is the first batch entered into the pipeline, the information about the type of input data 
            is analyzed and the different pipeline parameters are set (size of the images, labels, bits per pixel..)'''
    def __call__(self, batch_data, visualize=False):
        if visualize: original = [d.copy() for d in batch_data] #copy the original batch to diplay on visualization
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
                p_data, self.info_data = preprocess_dict_data_and_data_info(data)
                matrix = get_compose_matrix_and_configure_parameters(self.geom_ops, self.info_data)
                p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
                p_data['points_matrix'] = self.apply_geometry_transform_points(p_data['points_matrix'], matrix)
                self.process_data.append(p_data)

            for data in batch_data:
                lut = get_compose_function(self.color_ops)
                data['image'] = cv2.LUT(data['image'], lut)
                for op in self.indep_ops: data['image'] = op._apply_to_image_if_probability(data['image'])
                p_data = preprocess_dict_data(data)
                matrix = get_compose_matrix(self.geom_ops)
                p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
                p_data['points_matrix'] = self.apply_geometry_transform_points(p_data['points_matrix'], matrix)
                self.process_data.append(p_data)
        else:
            if self.info_data is None:  # First iteration to configure parameters and scan data info while the first item is being processed
                data = batch_data[0]
                batch_data = batch_data[1:]  # exclude the first item in the batch to be processed on the second loop
                lut = get_compose_function(self.color_ops)
                data['image'] = cv2.LUT(data['image'], lut)
                for op in self.indep_ops: data['image'] = op._apply_to_image_if_probability(data['image'])
                p_data, self.info_data = preprocess_dict_data_and_data_info_with_resize(data, new_size=self.resize)
                matrix = get_compose_matrix_and_configure_parameters(self.geom_ops, self.info_data)
                p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
                p_data['points_matrix'] = self.apply_geometry_transform_points(p_data['points_matrix'], matrix)
                self.process_data.append(p_data)

            for data in batch_data:
                lut = get_compose_function(self.color_ops)
                data['image'] = cv2.LUT(data['image'], lut)
                for op in self.indep_ops: data['image'] = op._apply_to_image_if_probability(data['image'])
                p_data = preprocess_dict_data_with_resize(data, self.info_data)
                matrix = get_compose_matrix(self.geom_ops)
                p_data['data_2d'] = self.apply_geometry_transform_data2d(p_data['data_2d'], matrix)
                p_data['points_matrix'] = self.apply_geometry_transform_points(p_data['points_matrix'], matrix)
                self.process_data.append(p_data)
        if visualize:
            return postprocess_data_and_visualize(self.process_data, original, self.info_data)
        else:
            return postprocess_data(self.process_data, self.info_data)


import numpy as np
import cv2
import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

img: np.ndarray = cv2.imread('../oso.jpg', )
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

segmap = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.int32)
segmap[28:171, 35:485, 0] = 1
'''segmap[10:25, 30:45, 0] = 2
segmap[10:25, 70:85, 0] = 3
segmap[10:110, 5:10, 0] = 4
segmap[118:123, 10:110, 0] = 5'''
#segmap = SegmentationMapsOnImage(segmap, shape=img.shape)

segmap2 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.int32)
segmap2[0:150, 50:125, 0] = 1
'''segmap[10:25, 30:45, 0] = 2
segmap[10:25, 70:85, 0] = 3
segmap[10:110, 5:10, 0] = 4
segmap[118:123, 10:110, 0] = 5'''
#segmap2 = SegmentationMapsOnImage(segmap2, shape=img.shape)

x = np.random.randn(img.shape[0]//4 * img.shape[1]//4)
y = np.random.randn(img.shape[0]//4 * img.shape[1]//4)

# Create heatmap
heatmap, xedges, yedges = np.histogram2d(x, y, bins=(img.shape[0]//4, img.shape[1]//4))
heatmap = heatmap / 3
heatmap_complete = np.zeros((img.shape[0], img.shape[1], 1))

heatmap_complete[0:img.shape[0]//4, 0:img.shape[1]//4, 0] = heatmap

keypoints = ([img.shape[0] // 2, img.shape[1] // 2], [img.shape[0] // 2 + 15, img.shape[1] // 2 - 50],
             [img.shape[0] // 2 + 85, img.shape[1] // 2 - 80], [img.shape[0] // 2 - 105, img.shape[1] // 2 + 60])

points = [torch.from_numpy(np.asarray(point)) for point in keypoints]
# data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW


# data = color.equalize_histogram(data, visualize=True)
data = {'image': img, 'mask': segmap2, 'mask2': segmap ,'keypoints': points, 'label': 5, 'heatmap': heatmap_complete}
samples = 10

batch = [data.copy() for _ in range(samples)]
batch2 = [data.copy() for _ in range(samples)]

from time import time



start_time = time()
pip = pipeline(  pipeline_operations=(
    translate_pipeline(probability=0.1, translation=(3, 1)),
    vflip_pipeline(probability=0.3),
    hflip_pipeline(probability=0.3),
    contrast_pipeline(probability=0, contrast_factor=1),
    random_brightness_pipeline(probability=1, brightness_range=(1, 1.2)),
    gamma_pipeline(probability=0, gamma_factor=0),
    random_translate_pipeline(probability=0, translation_range=(-90,90)),
    random_scale_pipeline(probability=0.5, scale_range=(0.5, 1.5), center_desviation=20),
    random_rotate_pipeline(probability=0, degrees_range=(-50, 50), center_desviation=20),
    random_translate_pipeline(probability=0, translation_range=(20, 100)),
    random_shear_pipeline(probability=0, shear_range=(0, 0.5))
))

batch = pip(batch, visualize=True)

operations = (brightness_pipeline(probability=1, brightness_factor=150),
              contrast_pipeline(probability=1, contrast_factor=2))

get_compose_function(operations)
consumed_time = time() - start_time
print(consumed_time)
print(consumed_time / samples)
