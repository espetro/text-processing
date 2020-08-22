from .search import AStar, JumpPoint
from scipy.signal import find_peaks
from numpy import ndarray
from tqdm import tqdm
from enum import Enum

import numpy as np
# import peakutils
import logging
import cv2

logging.basicConfig(level=logging.INFO)

class SearchAlg(Enum):
    ASTAR = 1
    JPS = 2

class LineSegmentation:
    """Represents the Line Segmentation algorithm.

    Parameters
    ----------
    image: numpy ndarray
        A binarized gray image with shape (height, width)
    """
    def __init__(self, image: ndarray):
        self.logger = logging.getLogger("lineSegm")

        if len(image.shape) == 3:
            # raise ValueError("Only accepting 2D gray images")
            self.logger.warning(" Only accepting 2D gray images")
            image = image[:,:,0]  # all three channels should be grayscale

        self.img = image
        self.height, self.width = image.shape[:2]

        self.lines = []


    def find_lines(self, alg: SearchAlg):
        """Find the lines"""
        ## Step 0: enhance the image (only used to find peaks & valleys)
        image = LineSegmentation.enhance(self.img)

        ## Step 1: Projection Analysis (find valleys / spaces, and peaks / text)
        peaks = LineSegmentation.find_peaks(image)
        self.logger.info(f" {len(peaks)} peaks detected.")

        valleys = LineSegmentation.find_valleys(peaks)
        self.logger.info(f" {len(valleys)} lines detected.")
        
        ## Step 2: Obtain the text lines with path planning
        Searcher = {SearchAlg.ASTAR: AStar, SearchAlg.JPS: JumpPoint}.get(alg)
        
        results = []
        for line in tqdm(valleys):
            start = (line, 0)
            goal = (line, self.width - 1)

            path_finder = Searcher(self.img)
            path = path_finder.find(start, goal)
            results.append((line, path))

        return results

        

    @staticmethod
    def minmax_scale(grid: ndarray):
        """Rescales an image grid to range 0..1"""
        mn, mx = grid.min(), grid.max()
        return (grid - mn) / (mx - mn)

    @staticmethod
    def enhance(image: ndarray):
        # invert
        image = image.max() - image
        # erode
        # kernel = np.ones((2,2), np.uint8)
        # image = cv2.erode(image.astype("uint8"), kernel, iterations=1)
        # dilate
        # kernel = np.ones((5,5), np.uint8)
        # image = cv2.dilate(image, kernel, iterations=1)
        return image

    @staticmethod
    def find_valleys(peaks):
        dist = lambda peaks, i: (peaks[i + 1] - peaks[i]) / 2

        return [peaks[i] + dist(peaks, i) for i in range(0, len(peaks) - 1)]
        
    @staticmethod
    def find_peaks(img: ndarray):
        # compute the ink density histogram (sum each rows)
        hist = cv2.reduce(img.astype("float64"), 1, cv2.REDUCE_SUM).ravel()
        # hist = np.sum(img, axis=1).ravel()
        
        logging.info(f" img: {img.shape}, hist: {hist.shape}")

        # find peaks withing the ink density histogram
        max_hist = hist.max()
        mean_hist = np.mean(hist)
        thres_hist = mean_hist / max_hist

        # peaks = peakutils.indexes(hist, thres=thres_hist, min_dist=50)
        peaks, _ = find_peaks(hist, threshold=None, distance=50)
        logging.info(f" Found {len(peaks)} original peaks.")

        # find peaks that are too high
        mean_peaks = np.mean(hist[peaks])
        std_peaks = np.std(hist[peaks])
        thres_peaks_high = mean_peaks + 1.5 * std_peaks
        thres_peaks_low = mean_peaks - 3 * std_peaks
        
        # Just return the peaks around the center of the peak distribution (remove outliers)
        _filter = np.logical_and(
            hist[peaks] > thres_peaks_low,
            hist[peaks] < thres_peaks_high
        )
        return peaks[_filter]

    
