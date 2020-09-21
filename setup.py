from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ida_lib',
    version='1.0',
    description='Image Data Augmentation library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/raquelvilas18/ida_lib',
    author='Raquel Vilas',
    author_email='raquel.rodriguez.vilas@rai.usc.es',
    packages=find_packages(where="ida_lib", exclude=["tests"]),
    python_requires=">=3.5",
    keywords=["Pytorch", "Image Data Augmentation"],
    install_requires=["opencv-python",
                      "numpy>=1.18.0",
                      "bokeh>=2.0.0",
                      "kornia>=0.3.1",
                      "torch>=1.5.0",
                      "pandas",
                      "tqdm"],
    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ]
)
