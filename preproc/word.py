
from skimage.filters import threshold_otsu
from numpy import ndarray, array

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import cv2

class WordSegmentation:
    """A class representing the word segmentation algorithm.
    In short, it uses the vertical projection of the binarized image to group whitespaces (blocks of pixels of value 255) and
    obtain maximal word bounding boxes (blocks of pixels of value 0)

    Source:
        https://github.com/muthuspark/ml_research/blob/master/Separate%20words%20in%20a%20line%20using%20VPP.ipynb
    """

    def __init__(self, image, line_bb):
        """
        Parameters
        ----------
            image: ndarray of shape (X,Y)
                binarized image of a line containing words
            line_bb: tuple (line start, line end)
                Tuple pointing to the starting and ending height of the line w.r.t. the original image
        """
        self.image = image
        self.height, self.width = image.shape[:2]
        self.lstart, self.lend = line_bb

        self.words = []

        # Compute the vertical projection of the image
        thresh = threshold_otsu(image)
        binary = image > thresh
        self.proj = np.sum(binary, axis=0)

    def get_words(self):
        """

        Returns
        -------
            words: list of (word, (x, y, height, width))
        """
        bbs = WordSegmentation.get_word_boxes(self.image, self.proj, self.height)

        self.words = ((self.image[:, ymin:ymax], (self.lstart, ymin, self.lend, ymax)) for (ymin, ymax) in bbs)
        self.words = [(word, params) for word, params in self.words if np.count_nonzero(word) != np.size(word)]  # remove all-white images
        return self.words

    def plot_projection(self):
        plt.figure()
        plt.plot(self.proj)
        plt.show()

    @staticmethod
    @nb.njit(cache=True)
    def get_word_boxes(image: ndarray, proj: array, height: int, min_ws_length=5, min_avg_ws_length=11, avg_factor=0.9, block_sz_factor=1.8):
        """"""

        # accumulate whitespace block sizes
        block_sizes = []
        block_sz = 0
        for px in proj:
            if px == height:
                block_sz +=1
            else:
                if block_sz > min_ws_length:
                    block_sizes.append(block_sz)
                block_sz = 0
        # sometimes, images end in whitespace (so the last block is not added) - add it manually
        block_sizes.append(block_sz)
        block_sizes = np.array(block_sizes)

        # compute extra data to get better boundaries
        # norm_sizes = np.round(block_sizes / len(proj), 3)
        
        # compute mean whitespace block size (and if the last block if way bigger than the others, drop it for the mean calc)
        upper_block_sz_avg = np.mean(block_sizes[:-1]) * 2
        if block_sizes[-1] > upper_block_sz_avg:
            block_sz_avg = np.mean(block_sizes[:-1])
        else:
            block_sz_avg = np.mean(block_sizes)

        # mean whitespace block size has to be greater than X=11
        block_sz_avg = max(min_avg_ws_length, block_sz_avg * avg_factor)

        # select whitespace blocks whose size is bigger than the mean
        divider_indexes = [0]
        block_sz = 0
        for index, px in enumerate(proj):
            if px == height:
                block_sz += 1
            elif block_sz > 0 and block_sz > block_sz_avg:
                new_index = index - int(block_sz / block_sz_factor)
                divider_indexes.append(new_index)
                block_sz = 0
                    
        # add the last one as well (the block factor here is way smaller, so that header lines dont keep a lot of whitespaces)
        new_index = index - int(block_sz / 1.1)
        divider_indexes.append(new_index)

        # return the bounding boxes
        indexes = np.array(divider_indexes)
        return np.column_stack((indexes[:-1], indexes[1:]))