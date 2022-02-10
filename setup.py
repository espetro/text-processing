"""
TFG Packages
------------

A package for keeping useful modules that I used in my bachelor thesis.

Links
-----
* `https://github.com/espetro/text-processing`

"""

from setuptools import setup

setup(
    name="tfgpkg",
    version="0.0.3",
    url="https://github.com/espetro/text-processing",
    license="BSD",
    author="Quim Terrasa",
    author_email="quino.terrasa+dev@gmail.com",
    description="A package for keeping useful modules that I used in my bachelor thesis.",
    py_modules=["tfgpkg"],
    install_requires=[
        "tensorflow == 2.5.3",
        "numpy",
        "editdistance",
        "opencv-python",
        "numba",
        "Pillow",
        "antlr4-python3-runtime == 4.7.2",
        "h5py",
        "pandas",
        "tqdm",
        "scikit-image",
        "scikit-learn == 0.22.2",
        "importlib_resources",
        "keras_octave_conv",
        "autocorrect == 2.1.2",
        "pyspellchecker",
        "langdetect"
    ],
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.7"
    ]
)