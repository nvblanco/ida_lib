import timeit
import random

import numpy as np

from ida_lib.core.pipeline import Pipeline
from ida_lib.core.pipeline_geometric_ops import HflipPipeline, VflipPipeline, RotatePipeline, ShearPipeline, \
    ScalePipeline, TranslatePipeline
from ida_lib.core.pipeline_local_ops import BlurPipeline, GaussianBlurPipeline, GaussianNoisePipeline, \
    PoissonNoisePipeline, SaltAndPepperNoisePipeline, SpekleNoisePipeline
from ida_lib.core.pipeline_pixel_ops import ContrastPipeline, BrightnessPipeline, GammaPipeline, NormalizePipeline, \
    DenormalizePipeline

operations_probability = 1
dtype = np.uint8
samples = 10
shape = (750, 750)

# double definition
pipeline_operations = [
            HflipPipeline(probability = operations_probability),
            VflipPipeline(probability = operations_probability),
            RotatePipeline(probability = operations_probability, degrees = 20),
            ShearPipeline(probability = operations_probability, shear = (shape[0] // 10, shape[1]//10)),
            ScalePipeline(probability = operations_probability, scale_factor = 1.2),
            TranslatePipeline(probability = operations_probability,translation = (shape[0] // 10, shape[1]//10)),
            ContrastPipeline(probability = operations_probability, contrast_factor = 1.2),
            BrightnessPipeline(probability = operations_probability, brightness_factor=1.3),
            GammaPipeline(probability=operations_probability, gamma_factor = 1),
            NormalizePipeline(probability=operations_probability),
            DenormalizePipeline(probability = operations_probability),
            HflipPipeline(probability = operations_probability),
            VflipPipeline(probability = operations_probability),
            RotatePipeline(probability = operations_probability, degrees = 20),
            ShearPipeline(probability = operations_probability, shear = (shape[0] // 10, shape[1]//10)),
            ScalePipeline(probability = operations_probability, scale_factor = 1.2),
            TranslatePipeline(probability = operations_probability,translation = (shape[0] // 10, shape[1]//10)),
            ContrastPipeline(probability = operations_probability, contrast_factor = 1.2),
            BrightnessPipeline(probability = operations_probability, brightness_factor=1.3),
            GammaPipeline(probability=operations_probability, gamma_factor = 1),
            NormalizePipeline(probability=operations_probability),
            DenormalizePipeline(probability = operations_probability)
    ]

def generate_images_batch(batchsize, shape):
    data =  {'image': np.random.randint(low=0, high=256, size=(shape[0], shape[1], 1), dtype=dtype)}
    return [data.copy() for _ in range(batchsize)]

def generate_no_dict_batch(batchsize, shape):
    data =  np.random.randint(low=0, high=256, size=(shape[0], shape[1], 1), dtype=dtype)
    return [data.copy() for _ in range(batchsize)]

def generate_image_and_keypoints_batch(batchsize,shape, num_points):
    data = {'image': np.random.randint(low=0, high=256, size=(shape[0], shape[1], 1), dtype=dtype),
             'keypoints': np.random.randint(1, shape[0], num_points * 2).reshape(num_points, 2)}
    return [data.copy() for _ in range(batchsize)]

def generate_image_and_masks_batch(batchsize,shape, num_masks):
    mask = np.zeros((shape[0],shape[1], 1), dtype=dtype)
    mask[0:50, 0:50] = 1
    data ={}
    for i in range(num_masks):
        name = 'mask'+str(i)
        data[name] = mask
    return [data.copy() for _ in range(batchsize)]

def get_n_operations(n):
    return random.sample(pipeline_operations, n)


def func_batchsize():
    samples = 10
    operations = get_n_operations(2)
    pip = Pipeline(interpolation='nearest',
                   pipeline_operations=operations)
    batch = generate_images_batch(1, shape)
    pip(batch) #avoid diferences in meassurements
    print("time according to batchsize\n------------------------------\nBatchsize\ttotal time\ttime per sample")
    for batchsize in [1,5,10,50,100,200]:
        t = 0
        for i in range(samples):
            batch = generate_images_batch(batchsize, shape)
            t += timeit.timeit(lambda:pip(batch), number = 1)
        t = t/samples
        print(str(batchsize) + '\t' + str(t) +'\t' + str(t/(batchsize)))


def func_resolution():
    samples = 10
    batchsize = 100
    operations = get_n_operations(5)
    pip = Pipeline(interpolation='nearest',
                   pipeline_operations=operations)
    batch = generate_images_batch(1, (1,1))
    pip(batch) #avoid diferences in meassurements
    print("time according to resolution\n------------------------------\nBatchsize\ttime per item\ttime per pixel")
    for shape in [(50,50),(100,100),(250,250), (500,500), (750,750)]:
        t = 0
        for i in range(samples):
            batch = generate_images_batch(batchsize,shape)
            t += timeit.timeit(lambda:pip(batch), number = 1)
        pixels = shape[0]*shape[1]
        t = t/samples
        print(str(shape) + '\t' + str(t/batchsize) + '\t' + str((t/batchsize)/pixels) )

def func_resolution_bilinear():
    samples = 10
    batchsize = 100
    operations = get_n_operations(5)
    pip = Pipeline(interpolation='bilinear',
                   pipeline_operations=operations)
    batch = generate_images_batch(1, (1,1))
    pip(batch) #avoid diferences in meassurements
    print("time according to resolution\n------------------------------\nBatchsize\ttime per item\ttime per pixel")
    for shape in [(50,50),(100,100),(250,250), (500,500), (750,750), (1000,1000)]:
        t = 0
        for i in range(samples):
            batch = generate_images_batch(batchsize,shape)
            t += timeit.timeit(lambda:pip(batch), number = 1)
        pixels = shape[0]*shape[1]
        t = t/samples
        print(str(shape) + '\t' + str(t/batchsize) + '\t' + str((t/batchsize)/pixels) )

def func_number_operations():
    samples = 10
    batchsize = 10
    print("time according to operations\n------------------------------\nBatchsize\ttime per item")
    for n_ops in range(1,15):
        operations = get_n_operations(n_ops)
        pip = Pipeline(interpolation='nearest',
                       pipeline_operations=operations)
        batch = generate_images_batch(batchsize, (5,5))
        pip(batch) #avoid diferences in meassurements
        t = 0
        for i in range(samples):
            batch = generate_images_batch(batchsize,shape)
            t += timeit.timeit(lambda:pip(batch), number = 1)
        t = t/samples
        print(str(n_ops) + '\t' + str(t/batchsize))

def func_keypoints():
    samples = 10
    batchsize = 10
    print("time according to operations\n------------------------------\nNum_points\ttime per item")
    for n_points in range(1,1000,30):
        operations = get_n_operations(5)
        pip = Pipeline(interpolation='nearest',
                       pipeline_operations=operations)
        batch =  generate_image_and_keypoints_batch(batchsize, shape, n_points)
        pip(batch) #avoid diferences in meassurements
        t = 0
        for i in range(samples):
            batch = generate_image_and_keypoints_batch(batchsize, shape, n_points)
            t += timeit.timeit(lambda:pip(batch), number = 1)
        t = t/samples
        print(str(n_points) + '\t' + str(t/batchsize))


def func_number_masks():
    samples = 1
    batchsize = 10
    print("time according to operations\n------------------------------\nNum masks\ttime per item")
    operations = get_n_operations(5)
    pip = Pipeline(interpolation='nearest',
                   pipeline_operations=operations)
    batch = generate_image_and_masks_batch(batchsize, shape, 1)
    pip(batch)  # avoid diferences in meassurements
    for n_masks in range(1, 11):
        t = 0
        for i in range(samples):
            batch = generate_image_and_masks_batch(batchsize, shape, n_masks)
            pip = Pipeline(interpolation='nearest',
                           pipeline_operations=operations)
            pip(batch)  # avoid diferences in meassurements
            batch = generate_image_and_masks_batch(batchsize, shape, n_masks)
            t += timeit.timeit(lambda: pip(batch), number=1)
        t = t / samples
        print(str(n_masks) + '\t' + str(t / batchsize))

for i in range(10):
    func_resolution_bilinear()


