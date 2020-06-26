
import numpy as np

class BoundingBox:
    """
    Image bounding boxes functions
    """
    @staticmethod
    def word_bbs_to_line(image, bbs):
        """
        Parameters
        ----------
            image: numpy array
                A 3D RGB or Monochromatic image
            bbs: list of lists
                A list of all word bounding boxes for the image
        Returns
        -------
            list of numpy arrays
                A list of all lines obtained by joining aligned words
        """
        pass

    @staticmethod
    def sort_words(bbs):
        """Given a list of word bounding boxes, sorts them from top to bottom, left to right."""
        pass
