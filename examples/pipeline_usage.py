"""In this file an example of how to use the idaLib pipeline is shown, in which you can see:
    * how to declare the pipeline
    * which format to use for the input elements
    * how to display or not the results
    * and how to execute it in general.

For more information see the documentation
"""

from time import time

import numpy as np

from ida_lib.core.pipeline import *
from ida_lib.core.pipeline_geometric_ops import *
from ida_lib.core.pipeline_local_ops import *

data_type = np.uint8

# Read the example image
img: np.ndarray = cv2.imread('../micky.jpg', )
# opencv read in format BGR but IDALib works on RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = img.astype('float32')  # Example of bits per pixel used

short_size = min(img.shape[0], img.shape[1])

# Generate an example of segmentation map over the image
segmap = np.zeros((img.shape[0], img.shape[1], 1), dtype=data_type)
segmap[28:171, 35:485, 0] = 1
segmap[10:25, 30:245, 0] = 2
segmap[10:25, 70:385, 0] = 3
segmap[10:110, 5:210, 0] = 4
segmap[18:223, 10:110, 0] = 5

# Generate 2 examples of masks
mask_example1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=data_type)
mask_example1[0:50, 0:50] = 1
mask_example2 = np.zeros((img.shape[0], img.shape[1], 1), dtype=data_type)
mask_example2[-150:, -50:] = 1

# Generate an example of heatmap over the image
x = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)
y = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)
# Create heatmap
heatmap, xedges, yedges = np.histogram2d(x, y, bins=(img.shape[0] // 4, img.shape[1] // 4))
heatmap_complete = np.zeros((img.shape[0], img.shape[1], 1))
heatmap_complete[0:img.shape[0] // 4, 0:img.shape[1] // 4, 0] = 1

# Generate an example of coordinates of keypoints over the image
# the list of keypoints can be expressed as an array of nx2 dimensions or as a list of arrays (1x2) (of 2 coordinates)
number_of_points = 20
# generate 20 random coordinates (to make sure they don't go outside the image boundaries set short_Size as the limit)
random_coordinates = np.random.randint(1, short_size, number_of_points * 2).reshape(number_of_points, 2)

# Generate the input item of the pipeline. Its very importantto name each element with its data type so that the
# pipeline understands them. *If the item contains more than one element of each type, just number them like mask1
# and mask2
data = {'image': img, 'keypoints': random_coordinates, 'mask1': mask_example1, 'mask2': mask_example2,
        'heatmap': heatmap_complete, 'target': 'mickey'}

# For this example we are going to use the same identical input element but repeated n times to create a batch so we
# can see the different transformations
samples = 10
batch = [data.copy() for _ in range(samples)]

start_time = time()  # time measurement

# Define the pipeline and operations.
pip = Pipeline(interpolation='nearest',
               pipeline_operations=(
                   ScalePipeline(probability=0.5, scale_factor=0.5),
                   ShearPipeline(probability=0.3, shear=(0.2, 0.2)),
                   TranslatePipeline(probability=0.4, translation=(10,50)),
                   HflipPipeline(probability=0.6, exchange_points=[(0, 5), (1, 6)]),
                   RandomRotatePipeline(probability=0.4, degrees_range=(-20, 20))))

# pass the batch through the pipeline and visualize the transformations
batch = pip(batch, visualize=True)

consumed_time = time() - start_time
# keep in mind that visualization is a significant overhead, so to take a good measure of
# performance set visualize=False
print("Total time consumed to process " + str(samples) + " samples: " + str(consumed_time))
print("Time per sample: :" + str(consumed_time / samples))
