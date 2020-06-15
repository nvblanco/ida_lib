Overview
======================

Functionalities
-----------------

The functionalities on which IdaLib focuses are:

1. Offering a **wide variety of image transformations**.
2. Allow the **joint transformation**  of:

         * image
         * heatmap
         * mask
         * segmaps
         * keypoints
         * any combination of them

3. Offer an **interactive visualization tool** that facilitates the debugging of programs.
4. Offer an **efficient operation composition pipeline** (parameterizable at probability and attribute level).
5. Offer a **Dataloader** object (as a generator) that integrates the pipeline and allows the supply of any neural network.
6. Offer a tool to perform **Image Data Augmentation directly to disk** in order to use the increased dataset directly in future situations


First steps
--------------


The central object of ida lib is its pipeline. To use it, just decide which PipelineOperations we want it to include. All other parameters are optional.

.. code-block:: Python

    example_pipipeline = Pipeline(pipeline_operations=(
                         ScalePipeline(probability=0.5, scale_factor=0.5),
                         ShearPipeline(probability=0.3, shear=(0.2, 0.2)),
                         TranslatePipeline(probability=0.4, translation=(10,50)),
                         HflipPipeline(probability=0.6, exchange_points=[(0, 5), (1, 6)]),
                         RandomRotatePipeline(probability=0.4, degrees_range=(-20, 20))
                         )
                         )

The pipelineOperations can be divided into 2 groups:

* the **classic operations**, where you indicate exactly the parameters of the operation (for example ```RotatePipeline(degrees=20)``` ). In [transformations](#operations) you can see what each one of them does
* and the **Random pipelineOPerations**, where what you define is a range of possible parameters, and each time the operation is applied it will take a different value within this range (``RandomRotatePipeline(degrees_range=(-20,30)``)

Once you have defined your pipeline, you can pass through it your batch data to be transformed.

.. warning::
    Remember that the entry for the pipeline must be composed of Python dicts. To be treated correctly, each dict's element must be associated with its type (images with 'image1', 'image2'...; masks with 'mask1', 'mask67'...):

.. code-block:: Python

    data1 = {'image': img1, 'keypoints': random_coordinates, 'mask': mask_example1}
    data2 = {'image': img2, 'keypoints2': random_coordinates2, 'mask': mask_example2}
    data3 = {'image': img3, 'keypoints3': random_coordinates3, 'mask': mask_example3}

    batch = [data1, data2, data3]


Finally we run the pipeline as many times as necessary:

.. code-block:: Python

    transformed_batch = example_pipipeline(batch)


Visualization tool
-------------------

Ida Lib includes a tool to visualize the transformations to facilitate code debugging.
It is an interactive tool developed with the bokeh framework and allows the selection of the data to be displayed in the image.

* The color code is used to differentiate each element and identify it in all the images.
* The dots are numbered in order to see their order
* Allows to compare different transformations obtained by the pipeline
* It also includes the targets in the visualization in order to have a complete view of the elements

.. image:: visualization.png
   :width: 120%
   :align: center

.. note::
    To test the visualization tool you can try this example:
    https://github.com/raquelvilas18/ida_lib/blob/master/examples/pipeline_usage.py

.. warning::
    For the visualization tool a bokeh server is deployed; therefore it is only possible to have one open execution.
    It is important to close previous runs in order to open new windows
