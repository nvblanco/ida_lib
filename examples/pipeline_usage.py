import numpy as np
from ida_lib.core.pipeline import *
from ida_lib.core.pipeline_geometric_ops import  *
from ida_lib.core.pipeline_local_ops import  *


img: np.ndarray = cv2.imread('../micky.jpg', )
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

segmap = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.int32)
segmap[28:171, 35:485, 0] = 1
segmap[10:25, 30:245, 0] = 2
segmap[10:25, 70:385, 0] = 3
segmap[10:110, 5:210, 0] = 4
segmap[18:223, 10:110, 0] = 5
# segmap = SegmentationMapsOnImage(segmap, shape=img.shape)

segmap2 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.int32)
segmap2[0:150, 50:125, 0] = 1
segmap2[10:25, 30:45, 0] = 2
segmap2[10:25, 70:85, 0] = 3
segmap2[10:110, 5:10, 0] = 4
segmap2[118:123, 10:110, 0] = 5
# segmap2 = SegmentationMapsOnImage(segmap2, shape=img.shape)

x = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)
y = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)

# Create heatmap
heatmap, xedges, yedges = np.histogram2d(x, y, bins=(img.shape[0] // 4, img.shape[1] // 4))
heatmap = heatmap / 3
heatmap_complete = np.zeros((img.shape[0], img.shape[1], 1))

heatmap_complete[0:img.shape[0] // 4, 0:img.shape[1] // 4, 0] = heatmap

keypoints = ([img.shape[0] // 2, img.shape[1] // 2], [img.shape[0] // 2 + 15, img.shape[1] // 2 - 50],
             [img.shape[0] // 2 + 85, img.shape[1] // 2 - 80], [img.shape[0] // 2 - 105, img.shape[1] // 2 + 60])

points = [torch.from_numpy(np.asarray(point)) for point in keypoints]
# data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW


# data = color.equalize_histogram(data, visualize=True)
#data = {'image': img, 'mask': segmap2, 'mask2': segmap, 'keypoints': points, 'label': 5, 'heatmap': heatmap_complete}
data = {'image': img, 'mask': segmap, 'keypoints': points, 'heatmap': heatmap_complete}
samples = 20

batch = [data.copy() for _ in range(samples)]
batch2 = [data.copy() for _ in range(samples)]

from time import time

start_time = time()

pip = pipeline(interpolation='bilinear', pipeline_operations=(
    ScalePipeline(probability=0, scale_factor=0.5),
    RotatePipeline(probability=0, degrees=40),
SpekleNoisePipeline(probability=1)))

batch = pip(batch, visualize=True)
batch2 = pip(batch2, visualize=False)

consumed_time = time() - start_time
print(consumed_time)
print(consumed_time / (samples * 2))
