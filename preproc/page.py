from preproc import Deslanter, LineSegmentation, WordSegmentation
from typing import Tuple, Union
from io import BufferedIOBase
from numpy import ndarray

import numpy as np
import cv2

class Page:
    PAGE_SIZE = (0,0)
    PARAGRAPH_SIZE = (0,0)
    LINE_SIZE = (0,0)
    WORD_SIZE = (0,0)

    def __init__(self, file: Union[str, BufferedIOBase], crop=False):
        self.words,  = [], []

        if isinstance(file, str):
            self.image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        else:
            self.image = file
        
        # Pre-process the image
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        gray = Page.preprocess(gray)

        if crop:
            self.image = Page.crop(self.image, gray)
        # Obtain lines
        self.lines = Page.get_lines(self.image)
        # Obtain words
        self.words = Page.get_words(self.lines)

    @staticmethod
    def resize(image: np.ndarray, sz: Tuple[int]):
        """Resizes the image to a new size. It uses different methods for zooming and shreding."""
        orig_sz = np.product(image.shape)
        new_sz = np.product(sz)
        if new_sz > orig_sz:
            method = cv2.INTER_LANCZOS4
        else:
            method = cv2.INTER_AREA

        return cv2.resize(image, sz, method)

    @staticmethod
    def crop(image, gray):
        """Crops the image to match paper area. Note that the image must have the 4 paper corners.

        Parameters
        ----------
            image: ndarray of shape (X,Y,3)
                The actual image to crop.

            gray: ndarray of shape (X,Y)
                Grayscale pre-processed image reference, used to compute the cropping area.
        """
        conf = dict(hello=1)
        return image

    @staticmethod
    def preprocess(gray):
        """A set of pre-processing steps (histogram equalization, adjust sigmoid, etc.)"""
        return gray

    @staticmethod
    def get_lines(image, scale_factor=2.0):
        """Extract a list of text lines given a binarized image.

        Parameters
        ----------
            image: ndarray of shape (height, width) and values [0, 255] or [0, 1]
        Returns
        -------
            lines: list of (image, [height, width])
        """
        return LineSegmentation(image, scale_factor).segment()

    @staticmethod
    def get_words(lines):
        return [WordSegmentation(line, (start, end)).get_words() for line, (start, end) in lines]

    @staticmethod
    def deslant_text(image):
        return Deslanter.deslant_image(image, pad_size=10)
