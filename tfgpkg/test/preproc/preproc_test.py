"""
`tfgpkg.preproc` contains the following functionalities:
    * A binarization module:
        * SauvolaBinarizer:
        * IlluminationBinarizer:
    * SegmentationNetwork (TBD)
    * A text line preprocessing algorithm:
        * Deslanter
        * Deskewer (TBD)
        * LineSegmentation, which relies on an A* method
    * A word preprocessing algorithm:
        * Quantize:
        * WordSegmentation
    * Page:

Test the following scenarios in `tfgpkg.preproc`. Most scenarios contain qualitative tests.
"""

# from tfgpkg.languages import LanguageTransformer, HTMLMinidownColorListener
from io import StringIO
import pkg_resources
import pytest
import os

def test_illumination_binarization_image():
    """Scenario: Given a RGB image, retrieve a binary image that is similar to the expected binary image in between a
    10..20%."""
    pass

def test_sauvola_binarization_image():
    """Scenario: Given a RGB image, retrieve a binary image that is similar to the expected binary image in between a
    10..20%. It must use Sauvola algorithm."""
    pass

def test_segment_image_into_paragraphs():
    """Scenario: Given a binary image with text in paragraphs, retrieve a set of paragraphs as cropped images."""
    pass

def test_segment_paragraph_into_lines():
    """Scenario: Given a binary image containing a paragraph, retrieve a set of lines, which must match the lines in the
    paragraph in the same order. Each line should not contain parts of characters from other lines (e.g. the tail of 'g'
    character from other line)."""
    pass

def test_deslant_line_image():
    """Scenario: Given a binary image containing a text line, retrieve a deslanted text line image."""
    pass

def test_deskew_line_image():
    """Scenario: Given a binary image containing a text line, retrieve a deskewed text line image."""
    pass

def test_quantize_word_image():
    """Scenario: Given a RGB word image, retrieve a quantized version of it in RGB having at most K=3 colors. These
    colors must not differ greatly from the original representation."""
    pass

def test_segment_line_into_words():
    """Scenario: Given a binary image containing a text line, retrieve a list of words as cropped images. The word order
    must be the same as in the input image, and at least 66% of the word images must contain at most 1 word."""
    pass

def test_paragraph_into_words():
    """Scenario: Given a RGB text image, retrieve a set of paragraphs, each having a set of lines which in turn have a
    set of words as cropped images. Displaying the words in hierarchical order (i.e. (word for line in paragraph)) must
    show the same order as in the original image."""
    pass