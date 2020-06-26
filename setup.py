"""
TFG Packages
------------

A module for storing util functions for my bachelor thesis.

Links
-----
* `https://github.com/espetro/text-processing`

"""

from setuptools import setup

setup(
    name="tfgpkg",
    version="0.0.1",
    url="https://github.com/espetro/text-processing",
    license="BSD",
    author="Quim Terrasa",
    author_email="quino.terrasa+dev@gmail.com",
    description="A module for storing util functions for my bachelor thesis.",
    py_modules=["preproc", "textRecognition", "languages"],
    install_requires=[
        "numpy",
        "editdistance",
        "opencv-python",
        "numba",
        "Pillow",
        "antlr4-python3-runtime",
        "h5py",
        "pandas",
        "keras >= '2.3.1'",
        "tqdm",
        "scikit-image",
        "scikit-learn",
        "importlib_resources",
        "keras_octave_conv",
        "mxnet",
        "jamspell",
        "pyspellchecker",
        "langdetect"
    ],
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6"
    ]
)