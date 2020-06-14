import os
import pathlib

import pytest

from ida_lib.core.pipeline_geometric_ops import RandomScalePipeline, HflipPipeline
from ida_lib.core.pipeline_pixel_ops import RandomContrastPipeline
from ida_lib.image_augmentation.augment_to_disk import AugmentToDisk


def count_files_in_directory(directory):
    count = 0
    for path in pathlib.Path(directory).iterdir():
        if path.is_file():
            count += 1
    return count


def reset_directory(directory):
    os.system("rm -rf " + directory)
    '''try:
        shutil.rmtree(directory)
    except OSError as e:
        print("Error: %s : %s" % (directory, e.strerror))'''


@pytest.mark.parametrize(
    ["samples_per_item"], [[1], [2], [5], [10]]
)
def test_augment_to_disk_work(dataset, samples_per_item):
    augmentor = AugmentToDisk(dataset=dataset,  # custom dataset that provides the input data
                              samples_per_item=samples_per_item,  # number of samples per input item
                              operations=(
                                  RandomScalePipeline(probability=0.6, scale_range=(0.8, 1.2), center_deviation=20),
                                  HflipPipeline(probability=0.5),
                                  RandomContrastPipeline(probability=0.5, contrast_range=(1, 1.5))),
                              interpolation='nearest',
                              padding_mode='zeros',
                              output_extension='.jpg',
                              output_csv_path='annotations.csv',
                              output_path='./test_augmented')
    augmentor()
    assert os.path.isdir('./test_augmented')
    number_elements = (len(dataset) * samples_per_item * 2)
    assert count_files_in_directory('./test_augmented') == number_elements
    reset_directory('./test_augmented')


@pytest.mark.parametrize(
    ["extension"], [['.jpg'], ['.png'], ['.jpeg']]
)
def test_output_extensions(dataset, extension):
    augmentor = AugmentToDisk(dataset=dataset,  # custom dataset that provides the input data
                              samples_per_item=2,  # number of samples per input item
                              operations=(
                                  RandomScalePipeline(probability=0.6, scale_range=(0.8, 1.2), center_deviation=20),
                                  HflipPipeline(probability=0.5),
                                  RandomContrastPipeline(probability=0.5, contrast_range=(1, 1.5))),
                              interpolation='nearest',
                              padding_mode='zeros',
                              output_extension=extension,
                              output_csv_path='annotations.csv',
                              output_path='./test_augmented')
    augmentor()
    contenido = os.listdir('./test_augmented')
    for i, elem in enumerate(contenido):
        if i == 5:
            break
        assert elem.split('.')[-1] == extension[1:]
    reset_directory('./test_augmented')
