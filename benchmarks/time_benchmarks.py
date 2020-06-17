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

shape = (50,50)

pipeline_operations = [
            HflipPipeline(probability = operations_probability),
            VflipPipeline(probability = operations_probability),
            RotatePipeline(probability = operations_probability, degrees = 20),
            ShearPipeline(probability = operations_probability, shear = (shape[0] // 10, shape[1]//10)),
            ScalePipeline(probability = operations_probability, scale_factor = 1.2),
            TranslatePipeline(probability = operations_probability,translation = (shape[0] // 10, shape[1]//10)),
            BlurPipeline(probability = operations_probability),
            GaussianBlurPipeline(probability = operations_probability),
            GaussianNoisePipeline(probability=operations_probability),
            PoissonNoisePipeline(probability = operations_probability),
            SaltAndPepperNoisePipeline(probability = operations_probability),
            SpekleNoisePipeline(probability= operations_probability),
            ContrastPipeline(probability = operations_probability, contrast_factor = 1.2),
            BrightnessPipeline(probability = operations_probability, brightness_factor=1.3),
            GammaPipeline(probability=operations_probability, gamma_factor = 1.2),
            NormalizePipeline(probability=operations_probability),
            DenormalizePipeline(probability = operations_probability)
    ]

def generate_images_batch(batchsize):
    data =  {'image': np.random.randint(low=0, high=256, size=(shape[0], shape[1], 3), dtype=dtype)}
    return [data.copy() for _ in range(batchsize)]

def get_n_operations(n):
    return random.sample(pipeline_operations, n)

''''TranslatePipeline',
           'RandomScalePipeline',
           'RandomRotatePipeline',
           'RandomShearPipeline',
           'RandomTranslatePipeline',
           'RandomContrastPipeline',
           'RandomBrightnessPipeline',
           'RandomGammaPipeline','''

pip = None
batch = None

def setup_pipeline(operations, batch_size):
    operations = get_n_operations(operations)
    pip = Pipeline(interpolation = 'nearest',
                   pipeline_operations = operations )
    batch = generate_images_batch(batch_size)
    return pip, batch


def func_batchsize():
    operations = get_n_operations(5)
    pip = Pipeline(interpolation='nearest',
                   pipeline_operations=operations)
    batch = generate_images_batch(1)
    pip(batch) #avoid diferences in meassurements
    for batchsize in [1,5,10,50,100]:
        t = 0
        for i in range(10):
            batch = generate_images_batch(batchsize)
            t += timeit.timeit(lambda:pip(batch), number = 1)
        print(timeit.timeit(lambda:pip(batch), number = 1)/batchsize)


func_batchsize()

"""if __name__ == "__main__":
    import timeit

    print(timeit.timeit("pip(batch)", setup = "from __main__ import setup_pipeline;"  + ', '.join(globals() + "setup_pipeline()", number=100)))"""
