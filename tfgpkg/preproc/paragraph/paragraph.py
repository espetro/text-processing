from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, OPTICS
from sklearn.neighbors import KernelDensity

from numpy import ndarray as NumpyArray
from sklearn.base import ClusterMixin
from typing import Tuple, List

import numpy as np

class ParagraphSegmentation:
    """Breaks a given text image into paragraphs, following the peak-valley model.
    This class requires a LineSegmentation instance to be ran previously, in order to obtain the valley lines
    that split text lines (or "peaks").
    """

    def __init__(self, coords: NumpyArray):
        self.coords = coords
        self.scaled_coords = StandardScaler().fit(coords).transform(coords)

    def get_lines_with_paragraphs(self, algorithm: str = "MeanShift") -> NumpyArray:
        if algorithm == "Kernel":
            labels = self._get_density_groups()
        else:
            labels = self._get_clusters(algorithm)

        return np.array(labels, dtype=np.uint8)

    def _get_density_groups(self):
        """For an in-depth explanation, please see:
            blog.mattoverby.net (1d-clustering-with-kde)
        """
        if False:
            points = np.array([np.mean(pos) for _, pos in self.lines]).reshape(-1, 1)
            points = StandardScaler().fit(points).transform(points)
            points = np.sort(points)

            estimator = KernelDensity(kernel="gaussian")
            estimator.fit(points)

            p_range = np.linspace(np.min(points), np.max(points)).reshape(-1, 1)
            p_distr = estimator.score_samples(p_range)
            p_min, p_max = argrelextrema(p_distr, np.less)[0], argrelextrema(p_distr, np.greater)[0]

        return np.ones((len(self.coords),)) * -1  # TBD

    def _get_clusters(self, algorithm):
        estimator: ClusterMixin = {
            "MeanShift": MeanShift(),
            "OPTICS": OPTICS(min_samples=2)
        }.get(algorithm)

        estimator.fit(self.scaled_coords)
        return estimator.labels_