from skimage.exposure import adjust_sigmoid
from skimage.filters import threshold_sauvola
from numpy import ndarray

import cv2
import numpy as np
import numba as nb

class FastBinarize:
    """
    An implementation of Binarize with a fast version of Sauvola Threshold.

    Sauvola threshold based in
            J. Sauvola, T. Seppanen, S. Haapakoski, M. Pietikainen,
            Adaptive Document Binarization, in IEEE Computer Society Washington, 1997.

    Source:
        https://github.com/arthurflor23/handwritten-text-recognition
        https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_sauvola

    This class uses numba to accelerate numpy-based functions
    """
    np.seterr(divide='ignore', invalid='ignore')

    @staticmethod
    def imread(fpath: str, grayscale=False):
        """Reads an image in grayscale mode"""
        if grayscale:
            return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_grayscale(image: ndarray, from_rgb=True):
        """Transforms an image from RGB to Grayscale"""
        if from_rgb:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def threshold_sauvola(image, window_sz=7):
        """Computes the sauvola thresholded image for the given image.

        Parameters
        ----------
            image: 3D numpy array
                A RGB image
            window_sz: int, default 3
                Window size for the Sauvola Threshold (look in source)

        Returns
        -------
            2D numpy array, the binarized result image
        """
        gray = FastBinarize.to_grayscale(image)
        w, h = gray.shape

        cei = FastBinarize.contrast_enhancement(gray, h, w)
        cei = cei.astype(np.uint8)
        
        threshold = threshold_sauvola(cei, window_size=window_sz)
        result = cei > threshold
        return result.astype(np.uint8) * 255  # from bool [0,1] to int [0, 255]

    @staticmethod
    @nb.njit()
    def contrast_enhancement(image: ndarray, h:int, w:int, c:float=0.3):
        """Computes the Contrast-Enhanced Image (CEI)"""
        # Get histogram
        bins = np.arange(0, 300, 10)
        bins[26] = 255
        hist = np.histogram(image, bins)
        
        # Histogram reducing value (hr)
        sqrt_hw = np.sqrt(h * w)
        hr = 0
        for i in range(len(hist[0])):
            if hist[0][i] > sqrt_hw:
                hr = i * 10
                break

        # Compute contrast-enhanced image (CEI)
        cei = (image - (hr + 50 * c)) * 2
        for x in range(h):
            for y in range(w):
                # keep values between 0..255
                if cei[x, y] > 255:
                    cei[x, y] = 255
                elif cei[x, y] < 0:
                    cei[x, y] = 0

        return cei
