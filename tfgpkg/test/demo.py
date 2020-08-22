"""
This test suite runs integration tests for checking the right behavior of the whole app / package.
"""

import pytest

def test_rgb_image_segment_into_words_and_recognized():
    """Scenario: Given a RGB image containing a text paragraph, retrieve a string that matches the paragraph's content
    or does not differ from it greatly."""
    pass

def test_rgb_image_segment_into_words_and_extracted_colors():
    """Scenario: Given a RGB image containing a text paragraph, retrieve a list of color tuples that matches the
    paragraph's word colors or does not differ from it greatly."""
    pass

def test_rgb_image_segment_into_words_recognized_and_colors():
    """Scenario: Given a RGB image containing a text paragraph, retrieve a string that matches the paragraph's content
    or does not differ from it greatly, and a list of color tuples of the same length, that matches the word colors or
    does not differ from them greatly."""
    pass

def test_rgb_image_segment_into_words_recognized_and_parsed():
    """Scenario: Given a RGB image containing a text paragraph, retrieve an HTML output containing a string that matches
    the paragraph content or does not differ from it greatly."""
    pass