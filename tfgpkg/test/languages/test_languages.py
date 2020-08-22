"""
Test the following scenarios in `tfgpkg.languages`:

Scenario: Given an empty input, retrieve an empty HTML output

Scenario: Given a badly-formatted input, raise an InputError

Scenario: Given a valid word, retrieve an HTML output with a <p>..</p> tag

Scenario: Given a valid sentence surrounded by '$', retrieve an HTML output with a <strong>..</strong> tag

Scenario: Given a valid sentence surrounded by '@', retrieve an HTML output with a <em>..</em> tag

Scenario: Given a valid sentence with a leading '#', retrieve an HTML output with a <h1>..</h1> tag

Scenario: Given a valid sentence whose first string is at least one or more '#' characters, retrieve an HTML output
    with a <hX>..</hX> tag, where X is the number of '#' characters

Scenario: Given a valid sentence with a color different from black, retrieve an HTML output with a coloured
    <span>..</span> tag

Scenario: Given a valid set of sentences, retrieve an HTML output where each sentence is displayed with its appropiate
    tag, and they are separated by newlines.

Scenario: Given a valid set of sentences whose first sentence has a leading '#' string, retrieve an HTML output with
    a <h1>..</h1> tag followed by other HTML tags. This set of sentence must contain colors different from black, and
    these must be properly reflected in the HTML output as <span>..</span> tags.

All scenarios must deal with UTF-8 strings. Available colors are listed at _________.
"""

from tfgpkg.languages import LanguageTransformer, HTMLMinidownColorListener
from io import StringIO
import pkg_resources
import pytest
import os

def read_from(fpath=None, text=None, out=None):
    """Returns a LanguageTransformer given an input (filepath or string) and an output (file object)"""
    return LanguageTransformer(HTMLMinidownColorListener, fpath=fpath, text_input=text, output_file=out)

def extract_words(lines):
    """Given a valid MinidownColor UTF-8 string, extract the words"""
    output = ""
    for line in lines:
        words = [w.strip("(").strip("$").strip("@") for w in line.split(" ") if "(" in w]
        output += " ".join(words) + "\n"

    return output[:-1]  # drop the last ''

HTML = {
    "doc_start": "<html><head><title>Result</title><meta charset='UTF-8'/></head><body>",
    "p_start": "<p>",
    "p_end": "</p>",
    "em_start": "<em>",
    "em_end": "</em>",
    "bold_start": "<strong>",
    "bold_end": "</strong>",
    "doc_end": "</body></html>"
}

# ============ TEST CASES ============


def test_empty_input():
    pass
    # expected_out = HTML["doc_start"] + HTML["doc_end"]

    # with StringIO() as f:
    #     read_from(text="", out=f)
    #     assert f.getvalue() == expected_out, "Empty input does not produce empty output"

def test_invalid_input():
    with pytest.raises(Exception):
        read_from(text="(Hello, black, None)", out=os.devnull)

def test_word():
    fpath = pkg_resources.resource_filename("tfgpkg", "test/resources/languages/base.hmd")

    expected_out = "\ufeff{}{}Hello{}{}".format(HTML["doc_start"], HTML["p_start"], HTML["p_end"], HTML["doc_end"])

    with StringIO() as f:
        read_from(fpath, out=f)
        assert f.getvalue() == expected_out, "Not a valid <p> tag HTML expression"

def test_bold_sentence():
    pass

def test_cursive_sentence():
    pass

def test_header_sentence():
    pass

def test_X_header_sentence():
    pass

def test_color_sentence():
    pass

def test_set_sentences():
    pass

def test_set_color_sentences():
    pass