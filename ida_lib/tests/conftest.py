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

def image_np(dtype = np.uint8):
    return np.random.randint(low=0, high=256, size=(size[0], size[1], 3), dtype=dtype)

def segmap(img, dtype = np.uint8):
    segmap = np.zeros((img.shape[0], img.shape[1], 1), dtype = dtype)
    segmap[28:171, 35:485, 0] = 1
    segmap[10:25, 30:245, 0] = 2
    segmap[10:25, 70:385, 0] = 3
    segmap[10:110, 5:210, 0] = 4
    segmap[18:223, 10:110, 0] = 5
    return segmap


def points():
    number_of_points = 15
    return np.random.randint(1, size[0], number_of_points * 2).reshape(number_of_points, 2)

def mask(img, dtype = np.uint8):
    mask_example1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=dtype)
    mask_example1[0:50, 0:50] = 1
    return mask_example1

def heatmap(img, dtype = np.uint8):
    heatmap_complete = np.zeros((img.shape[0], img.shape[1], 1))
    heatmap_complete[0:img.shape[0] // 4, 0:img.shape[1] // 4, 0] = 1
    return heatmap_complete

@pytest.fixture
def numpy_image():
    return {'image': image_np()}


@pytest.fixture
def numpy_monochannel_image():
    return {'image': np.random.randint(low=0, high=256, size=(size[0], size[1], 1), dtype=np.uint8)}


@pytest.fixture
def numpy_image_batch():
    data = {'image': image_np()}
    batch = [data.copy() for _ in range(samples)]
    return batch


@pytest.fixture
def torch_image():
    return { 'image': kornia.image_to_tensor(image_np())}


@pytest.fixture
def numpy_float_all_elements_item():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    return {'image': img, 'keypoints': points(), 'segmap': segmap(img, dtype=np.float), 'mask': mask(img, dtype = np.float),
            'heatmap': heatmap(img, dtype = np.float)}


@pytest.fixture
def numpy_image_and_points_batch():
    data = {'image': image_np(), 'keypoints': points()}
    batch = [data.copy() for _ in range(samples)]
    return batch


@pytest.fixture
def numpy_float_all_elements_batch():
    img = np.random.randint(low=0, high=256, size=(100, 100, 3)).astype(np.float)
    data = {'image': img, 'keypoints': points(), 'segmap': segmap(img, dtype=np.float), 'mask': mask(img, dtype = np.float),
            'heatmap': heatmap(img, dtype=np.float)}
    batch = [data.copy() for _ in range(samples)]
    return batch


@pytest.fixture
def torch_float_all_elements_item():
    img = np.random.randint(low=0, high=256, size=(100, 100, 3)).astype(np.float)
    return {'image':  kornia.image_to_tensor(img),
            'keypoints': points(),
            'segmap': kornia.image_to_tensor(segmap(img, dtype = np.float)),
            'mask':  kornia.image_to_tensor(mask(img, dtype=np.float)),
            'heatmap': kornia.image_to_tensor(heatmap(img, dtype=np.float))}


@pytest.fixture
def numpy_item_without_image():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    return {'segmap': segmap(img, dtype=np.float), 'mask': mask(img, dtype = np.float)}


@pytest.fixture
def numpy_item_2_mask():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    return {'image': img, 'mask': mask(img, dtype = np.float), 'mask2': mask(img, dtype = np.float)}


@pytest.fixture
def numpy_batch_2_mask():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    data = {'image': img, 'mask': mask(img, dtype = np.float), 'mask2': mask(img, dtype = np.float)}
    batch = [data.copy() for _ in range(samples)]
    return batch


@pytest.fixture
def numpy_batch_without_image():
    img = np.random.randint(low=0, high=256, size=(size[0], size[1], 3)).astype(np.float)
    data = {'segmap': segmap(img, dtype = np.float), 'mask': mask(img, dtype = np.float)}
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
