<img src="https://github.com/raquelvilas18/ida_lib/blob/master/ida_lib/static/icon.png" alt="tittle" width="70%"/>
<div style='text-align: justify;'> 

**IDA LIB (Image Data Augmentation Library)** is a Python library to optimize the task of *Image Data Augmentation*. This tool allows you to convert your input data into a larger and more diverse one in an efficient, fast and easy way.
</div> 
Ida Lib allows a wide variety of operations to be performed in order to provide the greatest possible diversity to the input dataset. 

The library is optimized to perform operations in the most efficient way possible, thus reducing the overload on other processes (such as the training process of a neural network). In addition, it allows the **joint transformation of different types and combinations of data types** in a flexible and correct way, including the processing of:
* Images
* Poin's coordinates
* Masks
* Segmentation maps
* Heatmaps

## Features

* [**Multiple fast augmentations**](#operations) based on Kornia an OpenCV libraries
* **Flexible**
* Complete tool, includes support for tasks associated with Pytorch-based neural networks
  * includes **support for a dataloader to directly feed the neural network** including Image Data Augmentation tool
  * includes support for a **tool to directly perform the Image Data Augmentation to disk.** To be able to use the increased dataset later and independently of the platform
* Supports **multiple types of combined data** (images, heat maps, segmentation maps, masks and keypoints)
* Includes a [**visualization tool**](#visualization) to make easier program debugging and can see the transformation results

## Getting Started

 To use IdaLib in your projects you just need to install it:
 
### Installing

You can use pip to install Ida-Lib

```   pip install ida-lib ```

If you want to get the latest version of the code before it is released on PyPI you can install the library from GitHub:

``` pip install -U git+https://github.com/raquelvilas18/ida_lib```


### First steps
The central object of ida lib is its pipeline. To use it, just decide which PipelineOperations we want it to include. All other parameters are optional.

```
example_pipipeline = Pipeline(pipeline_operations=(
                         ScalePipeline(probability=0.5, scale_factor=0.5),
                         ShearPipeline(probability=0.3, shear=(0.2, 0.2)),
                         TranslatePipeline(probability=0.4, translation=(10,50)),
                         HflipPipeline(probability=0.6, exchange_points=[(0, 5), (1, 6)]),
                         RandomRotatePipeline(probability=0.4, degrees_range=(-20, 20))
                         )
                         )
```
The pipelineOperations can be divided into 2 groups:
* the classic operations, where you indicate exactly the parameters of the operation (for example ```RotatePipeline(degrees=20)``` ). In [transformations](#operations) you can see what each one of them does
* and the Random pipelineOPerations, where what you define is a range of possible parameters, and each time the operation is applied it will take a different value within this range (RandomRotatePipeline(degrees_range=(-20,30))

Once you have defined your pipeline, you can pass through it your batch data to be transformed. Remember that the entry for the pipeline must be composed of dictionary type data. For each element to be treated correctly it must be associated with its type (images with 'image1', 'image2'...; masks with 'mask1', 'mask67'...):
```
data1 = {'image': img1, 'keypoints': random_coordinates, 'mask': mask_example1}
data2 = {'image': img2, 'keypoints2': random_coordinates2, 'mask': mask_example2}
data3 = {'image': img3, 'keypoints3': random_coordinates3, 'mask': mask_example3}

batch = [data1, data2, data3]
```

Finally we run the pipeline as many times as necessary:

```
transformed_batch = example_pipipeline(batch)
```

## Documentation

You can see the whole project documentation here:

  https://ida-lib.readthedocs.io/en/latest/index.html


## <a name="operations">Transformations</a>

The transformations included in the library are:

1. **hflip**: horizontal flipping the image

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/hflip.png"  width="50%"/>

2. **vflip**: vertical flipping the image

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/vflip.png" alt="tittle" width="50%"/>

3. **Affine**:carry out the transformation expressed in the operation matrix

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/affine.png" alt="tittle" width="50%"/>

4. **Rotate**:rotate the image by the indicated degrees counterclockwise

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/rotate.png" alt="tittle" width="50%"/>

5. **Shear**: linear map that displaces each point in fixed direction, by an amount proportional to its signed distance from the line that is parallel to that direction and goes through the origin

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/shear.png" alt="tittle" width="50%"/>

6. **Scale**: scale the image by making it smaller or larger (crop equivalent)

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/scale.png" alt="tittle" width="50%"/>

7. **Translate**: moves the image pixels to the positions indicated on each axis

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/translate.png" alt="tittle" width="50%"/>

8. **Change gamma**: adjust image's gamma (luminance correction) .

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/gamma.png" alt="tittle" width="50%"/>

9. **Change contrast:**: change the image contrast.

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/contrast.png" alt="tittle" width="50%"/>


10. **Change brightness**: change the image brightness

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/brightness.png" alt="tittle" width="50%"/>

11. **Equalize histogram**: equalize the image histogram

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/equalization.png" alt="tittle" width="50%"/>

12. **Inject gaussian noise**: gaussian noise is a statistical noise having a probability density function (PDF) equal
to that of the normal distribution

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/gaussian_noise.png" alt="tittle" width="50%"/>

13. **Inject salt and pepper noise**: salt-and-pepper noise is a statistical noise compose of white (salt) and black (pepper) pixels

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/salt_pepper_noise.png" alt="tittle" width="50%"/>

14. **Inject spekle noise**: Speckle is a granular interference that inherently exists in and degrades the quality of the active radar,
synthetic aperture radar (SAR), medical ultrasound and optical coherence tomography images.
It is applied by adding the image multiplied by the noise matrix -> img + img * uniform_noise

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/spekle_noise.png" alt="tittle" width="50%"/>

15. **Inject poisson noise**: It is applied by adding Poisson-distributed noise

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/poisson_noise.png" alt="tittle" width="50%"/>

16. **Blur**: blur image.

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/blur.png" alt="tittle" width="50%"/>

17. **Gaussian blur**: blurring an image by a Gaussian function.

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/gaussian_blur.png" alt="tittle" width="50%"/>

##  <a name="visualization"> Visualization tool </a>

Ida Lib includes a tool to visualize the transformations to facilitate code debugging.
It is an interactive tool developed with the bokeh framework and allows the selection of the data to be displayed in the image.

* The color code is used to differentiate each element and identify it in all the images.
* The dots are numbered in order to see their order
* Allows to compare different transformations obtained by the pipeline
* It also includes the targets in the visualization in order to have a complete view of the elements

<img src="https://github.com/raquelvilas18/ida_lib/blob/master/docs/source/visualization.png" alt="tittle" width="100%"/>


## Built With

* [Pytorch](https://pytorch.org/) - The machine learning framework used
* [Bokeh](https://bokeh.org/) - The visualization library used 
* [Kornia](https://kornia.github.io/) - computer vision library that is used as a base for transformations
* [OpenCV](https://bokeh.org/) - computer vision library that is used as a base for transformations


## Authors

* **Raquel Vilas** - *programming effort*
* **Nicolás Vila Blanco** - *co-author*
* **Maria José Carreira Nouche** - *co-author*
