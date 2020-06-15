from ida_lib.core.pipeline import Pipeline
from ida_lib.core.pipeline_geometric_ops import ScalePipeline, ShearPipeline, HflipPipeline
from ida_lib.core.pipeline_local_ops import GaussianNoisePipeline
import numpy as np

from ida_lib.global_parameters import identity, one_torch, ones_torch, data_types_2d, restart_global_parameters
from ida_lib.operations.transforms import rotate
from ida_lib.operations.utils import arrays_equal

samples = 5
size= (100, 100)
def numpy_image_and_points_batch():
    number_of_points = 15
    random_coordinates = np.random.randint(1, size[0], number_of_points * 2).reshape(number_of_points, 2)
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3), dtype=np.uint8)
    data = {'image': img, 'keypoints': random_coordinates}
    batch = [data.copy() for _ in range(samples)]
    return batch

item = numpy_image_and_points_batch()
original_points = item[0]['keypoints'].copy()
pip = Pipeline(interpolation='nearest',
                   pipeline_operations=(
                       HflipPipeline(probability=1, exchange_points=((0,1), (3,4))),
                       HflipPipeline(probability=1)))
augmented = pip(item)
transformed_points = augmented[0]['keypoints'].cpu().numpy().astype(np.uint8)
print("arr1: {}, {} arr2: {} {}",original_points[0][0], original_points[0][1], transformed_points[0][0], transformed_points[0][1] )
assert arrays_equal(original_points[0], transformed_points[1])
assert arrays_equal(original_points[1], transformed_points[0])