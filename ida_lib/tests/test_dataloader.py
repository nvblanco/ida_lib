import pytest

from ida_lib.core.pipeline_geometric_ops import TranslatePipeline, RandomShearPipeline
from ida_lib.image_augmentation.data_loader import AugmentDataLoader


@pytest.mark.parametrize(
    ["batchsize"], [[1], [2], [3], [5], [10]]
)
def test_dataloader_work(dataset, batchsize):
    dataloader = AugmentDataLoader(dataset=dataset,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pipeline_operations=(
                                       TranslatePipeline(probability=1, translation=(30, 10)),
                                       RandomShearPipeline(probability=0.5, shear_range=(0, 0.5))),
                                   interpolation='bilinear',
                                   padding_mode='zeros'
                                   )
    for i_batch, sample_batched in enumerate(dataloader):  # our dataloader works like a normal dataloader
        assert 'image' in sample_batched
        assert 'mask' in sample_batched
        assert sample_batched['image'].shape[0] == batchsize
        if i_batch == 2:
            break


@pytest.mark.parametrize(
    ["resize"], [[(10, 10)], [(10, 50)], [(50, 10)], [(500, 500)]]
)
def test_dataloader_resize_work(dataset, resize):
    dataloader = AugmentDataLoader(dataset=dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   pipeline_operations=(
                                       TranslatePipeline(probability=1, translation=(30, 10)),
                                       RandomShearPipeline(probability=0.5, shear_range=(0, 0.5))),
                                   resize=resize,
                                   interpolation='bilinear',
                                   padding_mode='zeros'
                                   )
    for i_batch, sample_batched in enumerate(dataloader):  # our dataloader works like a normal dataloader
        assert 'image' in sample_batched
        assert sample_batched['image'].shape[2:] == resize
        if i_batch == 2:
            break
