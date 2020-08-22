"""
`tfgpkg.recognition` contains the following functionalities:
    * TinyData:
    * DataUnpack:
    * HighlightDetector
    * A color extraction algorithm:
        * ColorThief:
        * ColorGroup:
        * ColorExtractor:
    * A text recognition algorithm:
        * RecognitionNet, with an implementation of BaseModel
        * StringVectorizer
    * TextChecker:

Test the following scenarios in `tfgpkg.recognition`. Most scenarios contain qualitative tests. Scenarios with strings
must deal with UTF-8 strings. Available colors are listed at `tfgpkg.recognition.ColorGroup`.
"""

from io import BytesIO
from tempfile import NamedTemporaryFile

import tfgpkg.recognition as rec
import pkg_resources
import pytest
import os

# ============ TEST CASES ============

def test_df_images_to_h5():
    """Scenario: Given a valid set of train/test/validation image folders, and a valid DataFrame holding info. related
    to them, retrieve a `.h5` file containing the same hierarchical information as numpy arrays."""
    pass

def test_h5_to_np_images():
    """Scenario: Given a valid `.h5` file containing images and metadata, grouped in 'train', 'test' or 'valid' sets,
    retrieve a set of numpy arrays holding the same hierarchical information."""
    pass

def test_non_highlighted_image():
    """Scenario: Given a RGB non-highlighted word image, check if the word is recognized as is."""
    pass

def test_highlighted_image():
    """Scenario: Given a RGB highlighted word image, check if the word is recognized as is."""
    pass

def test_image_palette():
    """Scenario: Given a RGB word image with at least 2 main colors, check if the retrieved main colors do not differ
    greatly."""
    pass

def test_css_name_rgb():
    """Scenario: Given a CSS color name in RGB format, check if the retrieved CSS color name is the same or does not
    differ greatly."""
    pass

def test_image_palette_with_css_names():
    """Scenario: Given a RGB word image with at least 3 main colors, check if the retrieved main colors match the
    expected CSS color names or do not differ greatly."""
    pass

def test_vectorized_word():
    """Scenario: Given a word string, check if the retrieved CTC loss array matches the expected array."""
    pass

def test_unvectorized_word():
    """Scenario: Given a CTC loss array, check if the retrieved string matches the expected word."""
    pass

def test_spell_checked_sentence():
    """Scenario: Given a sentence string, check if the retrieved (corrected) sentence matches the expected corrected
    sentence or does not differ greatly."""
    pass

def test_word_recognition():
    """Scenario: Given a RGB binary image, check if the retrieved string matches the expected word or does not differ
    greatly."""
    pass

def test_coloured_word_recognition():
    """Scenario: Given a RGB binary image, check if the retrieved string matches the expected word or does not differ
    greatly, and the retrieved CSS font color is not black."""
    pass

def test_non_highlighted_word_recognition():
    """Scenario: Given a RGB binary image, check if the retrieved string matches the expected word or does not differ
    greatly, and the retrieved CSS background color is white."""
    pass

def test_highlighted_word_recognition():
    """Scenario: Given a RGB highlighted binary image, check if the retrieved string matches the expected word or does
    not differ greatly, and the retrieved CSS background color is not white."""
    pass
