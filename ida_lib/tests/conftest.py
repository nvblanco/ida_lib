import kornia
import numpy as np
import pytest
from torch.utils.data import Dataset

from ida_lib.core.pipeline import Pipeline
from ida_lib.core.pipeline_geometric_ops import ScalePipeline, ShearPipeline, TranslatePipeline, HflipPipeline, \
    RandomRotatePipeline
from ida_lib.core.pipeline_local_ops import GaussianNoisePipeline


size = (100, 100, 3)
samples = 10


@pytest.fixture
def numpy_image():
    return {'image': np.random.randint(low=0, high=256, size=(size[0], size[1], 3), dtype=np.uint8)}


@pytest.fixture
def numpy_monochannel_image():
    return {'image': np.random.randint(low=0, high=256, size=(size[0], size[1], 1), dtype=np.uint8)}


@pytest.fixture
def numpy_image_batch():
    data = {'image': np.random.randint(low=0, high=256, size=(size[0], size[1], 3), dtype=np.uint8)}
    batch = [data.copy() for _ in range(samples)]
    return batch


@pytest.fixture
def torch_image():
    return {
        'image': kornia.image_to_tensor(np.random.randint(low=0, high=256, size=(size[0], size[1], 3), dtype=np.uint8))}


@pytest.fixture
def numpy_float_all_elements_item():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    # Generate an example of segmentation map over the image
    segmap = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    segmap[28:171, 35:485, 0] = 1
    segmap[10:25, 30:245, 0] = 2
    segmap[10:25, 70:385, 0] = 3
    segmap[10:110, 5:210, 0] = 4
    segmap[18:223, 10:110, 0] = 5

    # Generate 2 examples of masks
    mask_example1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    mask_example1[0:50, 0:50] = 1
    # Generate an example of heatmap over the image
    x = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)
    y = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)
    # Create heatmap
    heatmap_complete = np.zeros((img.shape[0], img.shape[1], 1))
    heatmap_complete[0:img.shape[0] // 4, 0:img.shape[1] // 4, 0] = 1
    number_of_points = 15
    # generate 20 random coordinates (to make sure they don't go outside the image boundaries set short_Size as the
    # limit)
    random_coordinates = np.random.randint(1, size[0], number_of_points * 2).reshape(number_of_points, 2)

    return {'image': img, 'keypoints': random_coordinates, 'segmap': segmap, 'mask': mask_example1,
            'heatmap': heatmap_complete}


@pytest.fixture
def numpy_image_and_points_batch():
    number_of_points = 15
    random_coordinates = np.random.randint(1, size[0], number_of_points * 2).reshape(number_of_points, 2)
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3), dtype=np.uint8)
    data = {'image': img, 'keypoints': random_coordinates}
    batch = [data.copy() for _ in range(samples)]
    return batch


@pytest.fixture
def numpy_float_all_elements_batch():
    img = np.random.randint(low=0, high=256, size=(100, 100, 3)).astype(np.float)

    # Generate an example of segmentation map over the image
    segmap = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    segmap[28:171, 35:485, 0] = 1
    segmap[10:25, 30:245, 0] = 2
    segmap[10:25, 70:385, 0] = 3
    segmap[10:110, 5:210, 0] = 4
    segmap[18:223, 10:110, 0] = 5

    # Generate 2 examples of masks
    mask_example1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    mask_example1[0:50, 0:50] = 1
    # Generate an example of heatmap over the image
    x = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)
    y = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)
    # Create heatmap
    heatmap_complete = np.zeros((img.shape[0], img.shape[1], 1))
    heatmap_complete[0:img.shape[0] // 4, 0:img.shape[1] // 4, 0] = 1

    number_of_points = 15
    # generate 20 random coordinates (to make sure they don't go outside the image boundaries set short_Size as the
    # limit)
    random_coordinates = np.random.randint(1, 100, number_of_points * 2).reshape(number_of_points, 2)
    data = {'image': img, 'keypoints': random_coordinates, 'segmap': segmap, 'mask': mask_example1,
            'heatmap': heatmap_complete}
    batch = [data.copy() for _ in range(samples)]
    return batch


@pytest.fixture
def torch_float_all_elements_item():
    img = np.random.randint(low=0, high=256, size=(100, 100, 3)).astype(np.float)

    # Generate an example of segmentation map over the image
    segmap = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    segmap[28:171, 35:485, 0] = 1
    segmap[10:25, 30:245, 0] = 2
    segmap[10:25, 70:385, 0] = 3
    segmap[10:110, 5:210, 0] = 4
    segmap[18:223, 10:110, 0] = 5

    # Generate 2 examples of masks
    mask_example1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    mask_example1[0:50, 0:50] = 1
    # Generate an example of heatmap over the image
    x = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)
    y = np.random.randn(img.shape[0] // 4 * img.shape[1] // 4)
    # Create heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(img.shape[0] // 4, img.shape[1] // 4))
    heatmap_complete = np.zeros((img.shape[0], img.shape[1], 1))
    heatmap_complete[0:img.shape[0] // 4, 0:img.shape[1] // 4, 0] = 1

    number_of_points = 15
    # generate 20 random coordinates (to make sure they don't go outside the image boundaries set short_Size as the
    # limit)
    random_coordinates = np.random.randint(1, 100, number_of_points * 2).reshape(number_of_points, 2)
    img = kornia.image_to_tensor(img)
    segmap = kornia.image_to_tensor(segmap)
    mask_example1 = kornia.image_to_tensor(mask_example1)
    heatmap_complete = kornia.image_to_tensor(heatmap_complete)
    return {'image': img, 'keypoints': random_coordinates, 'segmap': segmap, 'mask': mask_example1,
            'heatmap': heatmap_complete}


@pytest.fixture
def numpy_item_without_image():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    # Generate an example of segmentation map over the image
    segmap = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    segmap[28:171, 35:485, 0] = 1
    segmap[10:25, 30:245, 0] = 2
    segmap[10:25, 70:385, 0] = 3
    segmap[10:110, 5:210, 0] = 4
    segmap[18:223, 10:110, 0] = 5

    # Generate 2 examples of masks
    mask_example1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    mask_example1[0:50, 0:50] = 1

    return {'segmap': segmap, 'mask': mask_example1}


@pytest.fixture
def numpy_item_2_mask():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    # Generate 2 examples of masks
    mask_example1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    mask_example1[0:50, 0:50] = 1
    mask_example2 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    mask_example2[50:90, 40:50] = 1
    return {'image': img, 'mask': mask_example1, 'mask2': mask_example2}


@pytest.fixture
def numpy_batch_2_mask():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    # Generate 2 examples of masks
    mask_example1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    mask_example1[0:50, 0:50] = 1
    mask_example2 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    mask_example2[50:90, 40:50] = 1
    data = {'image': img, 'mask': mask_example1, 'mask2': mask_example2}
    batch = [data.copy() for _ in range(samples)]
    return batch


@pytest.fixture
def numpy_batch_without_image():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    # Generate an example of segmentation map over the image
    segmap = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    segmap[28:171, 35:485, 0] = 1
    segmap[10:25, 30:245, 0] = 2
    segmap[10:25, 70:385, 0] = 3
    segmap[10:110, 5:210, 0] = 4
    segmap[18:223, 10:110, 0] = 5

    # Generate 2 examples of masks
    mask_example1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
    mask_example1[0:50, 0:50] = 1

    data = {'segmap': segmap, 'mask': mask_example1}
    batch = [data.copy() for _ in range(samples)]
    return batch


@pytest.fixture
def pipeline():
    pip = Pipeline(interpolation='nearest',
                   pipeline_operations=(
                       ScalePipeline(probability=1, scale_factor=0.5),
                       ShearPipeline(probability=0, shear=(0.2, 0.2)),
                       TranslatePipeline(probability=0, translation=(10, 50)),
                       HflipPipeline(probability=0, exchange_points=[(0, 5), (1, 6)]),
                       RandomRotatePipeline(probability=0, degrees_range=(-20, 20)),
                       GaussianNoisePipeline(probability=0)))
    return pip


@pytest.fixture
def empty_pipeline():
    pip = Pipeline(interpolation='nearest', pipeline_operations=())
    return pip


@pytest.fixture
def resize_pipeline():
    pip = Pipeline(interpolation='nearest',
                   resize=(10, 10),
                   pipeline_operations=(
                       ScalePipeline(probability=1, scale_factor=0.5),
                       ShearPipeline(probability=0, shear=(0.2, 0.2)),
                       TranslatePipeline(probability=0, translation=(10, 50)),
                       HflipPipeline(probability=0, exchange_points=[(0, 5), (1, 6)]),
                       RandomRotatePipeline(probability=0, degrees_range=(-20, 20)),
                       GaussianNoisePipeline(probability=0)))
    return pip

class DummyDataset(Dataset):
    def __init__(self, data_root=None, channels=1):
        self.data_root = data_root
        self.data_index = self.build_index(self.data_root)
        self.channels = channels

    @staticmethod
    def build_index(data_root):
        return range(10)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        sample = self.data_index[idx]

        # load data, NOTE: modify by cv2.imread(...)
        image = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
        mask = np.random.randint(low=0, high=1, size=(size[0], size[1], 3)).astype(np.float)

        return {'image': image, 'mask': mask}


@pytest.fixture
def dataset():
    return DummyDataset()
