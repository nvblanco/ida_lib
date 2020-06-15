.. IDA_LIB documentation master file, created by
   sphinx-quickstart on Mon Jun 15 11:25:21 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: icon.png
   :width: 70%
   :align: center

IdaLib
========================

IdaLib (Image Data Augmentation Library) is a library dedicated to the task of Image Data Augmentation in a fast, simple, efficient and flexible way
This library allows you to convert your input data into a larger and more diverse one to perform a better train of your Neural Network

Features
----------

* Multiple fast augmentations based on Kornia an OpenCV libraries
* Flexible
* Complete tool, includes support for tasks associated with Pytorch-based neural networks
* Supports multiple types of combined data (images, heat maps, segmentation maps, masks and keypoints)


Instalation
-----------

You can use pip to install Ida-Lib

``pip install ida-lib``

If you want to get the latest version of the code before it is released on PyPI you can install the library from GitHub:

``pip install -U git+https://github.com/raquelvilas18/ida_lib``

Contents
-----------

.. toctree::
   :maxdepth: 3

   overview
   transformations
   ida_lib
   examples



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Built with
-------------
* [Pytorch](https://pytorch.org/) - The machine learning framework used
* [Bokeh](https://bokeh.org/) - The visualization library used
* [Kornia](https://kornia.github.io/) - computer vision library that is used as a base for transformations
* [OpenCV](https://bokeh.org/) - computer vision library that is used as a base for transformations
* [Pycharm](https://rometools.github.io/rome/) - Development IDE

Acknowledgements
--------------------

* **Nicolás Vila Blanco** : project co-author
* **María José Carreira Nouche**: project co-author
* **CITiUS**: support company