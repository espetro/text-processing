# from .search import AStar, JumpPoint
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.grid import Grid
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
        
        self.peaks = []
        self.valleys = []
        self.lines = []


    def find_lines(self, alg: SearchAlg=SearchAlg.ASTAR):
        """Find the lines"""
        ## Step 0: enhance the image (only used to find peaks & valleys)ñ
        image = LineSegmentation.enhance(self.img)

        ## Step 1: Projection Analysis (find valleys / spaces, and peaks / text)
        self.peaks = LineSegmentation.find_peaks(image)
        self.logger.info(f" {len(self.peaks)} peaks detected.")

        self.valleys = LineSegmentation.find_valleys(self.peaks)
        self.logger.info(f" {len(self.valleys)} lines detected.")
                
        ## Step 2: If the valley splits the peaks with a straight line, then stop and return
        right_lines, wrong_lines = self.split_peaks_by_valleys(self.peaks, self.valleys)

        self.logger.info(f" Right lines: {right_lines}\nWrong lines: {wrong_lines}\n")

        if len(wrong_lines) == 0:
            return self.get_line_images(right_lines)
        else:
            ## Step 3: Otherwise, try to find a path to split both valleys
            grid = Grid(matrix=self.img)
            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            results = []

            for (start, end) in tqdm(wrong_lines):
                if sum(self.img[start, :]) != self.width - 1:
                    _start = grid.node(start, 0)
                    _end = grid.node(start, self.width - 1)
                    path, _ = finder.find_path(_start, _end, grid)
                    # start = LineSegmentation.process_path(path)

                if sum(self.img[end, :]) != self.width - 1:
                    _start = grid.node(end, 0)
                    _end = grid.node(end, self.width - 1)
                    path, _ = finder.find_path(_start, _end, grid)
                    # end = LineSegmentation.process_path(path)

                self.logger.info(f" Path: {path}")
                results.append((start, end))

            return self.get_line_images(right_lines + results)

    def split_peaks_by_valleys(self, peaks, valleys):
        """Checks if all the peaks can be separated by the valleys (drawing a straight line in the valleys).
        Returns the peaks that were possible to split and the peaks which were not."""
        right_lines, wrong_lines = [], []
        limited_valleys = [0, *valleys, self.height - 1]

        for (start, end) in zip(limited_valleys, limited_valleys[1:]):
            upper_line = self.img[start, :]
            lower_line = self.img[end, :]

            if (np.sum(upper_line) == self.width) and (np.sum(lower_line) == self.width):  # both lines are 100% white
                right_lines.append((start, end))
            else:
                wrong_lines.append((start, end))

        return (right_lines, wrong_lines)

    def get_line_images(self, lines):
        """Given a set of tuples (start, end), return the lines cut by these points in the image"""
        return [self.img[start:end, :] for start, end in lines] # it'd be great to remove big white areas prior to return

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

        return np.sort(np.array([peaks[i] + dist(peaks, i) for i in range(0, len(peaks) - 1)], np.uint8))
        
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
        return peaks[_filter].astype(np.uint8)

    @staticmethod
    def process_path(path):
        """Given a path of points, returns the average line that cross all of them"""
        return np.average((x for x, _ in path))
    
