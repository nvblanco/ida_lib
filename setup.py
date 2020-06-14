from setuptools import setup, find_packages

setup(
    name='ida_lib',
    version='1.0',
    description='Image Data Augmentation library',
    url='https://github.com/raquelvilas18/ida_lib',
    author='Raquel Vilas',
    author_email='raquel.rodriguez.vilas@rai.usc.es',
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.5",
    keywords="pytorch image data augmentation",
    install_requires = ["opencv-python",
                        "numpy>=1.18.4",
                        "bokeh>=2.0.1",
                        "kornia>=0.3.1",
                        "torch>=1.5.0",
                        "pandas",
                        "tqdm"],
    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        "Topic :: Software Development :: Libraries",
         "Topic :: Software Development :: Libraries :: Python Modules",
    ]
)