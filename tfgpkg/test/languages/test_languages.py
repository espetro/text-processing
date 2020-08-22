"""
Test the following scenarios in `tfgpkg.languages`:

Scenario: Given an empty input, raise an SyntaxError

Scenario: Given a badly-formatted input, raise an SyntaxError

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

HTML = HTMLMinidownColorListener.TAGS

# ============ TEST CASES ============


def test_empty_input():
    with pytest.raises(SyntaxError):
        read_from(text="", out=os.devnull)

def test_invalid_input():
    with pytest.raises(SyntaxError):
        read_from(text="(Hello, black, None)", out=os.devnull)

def test_word():
    fpath = pkg_resources.resource_filename("tfgpkg", "test/resources/languages/base.hmd")
    expected_out = "{}{}Hello {}{}".format(HTML["doc_start"], HTML["p_start"], HTML["p_end"], HTML["doc_end"])

    with StringIO() as f:
        read_from(fpath, out=f)
        assert f.getvalue() == expected_out, "Not a valid <p> tag HTML expression"

def test_bold_sentence():
    fpath = pkg_resources.resource_filename("tfgpkg", "test/resources/languages/zero.hmd")
    text = "this <strong>is a bold</strong> test. "
    expected_out = "{}{}{}{}{}".format(HTML["doc_start"], HTML["p_start"], text, HTML["p_end"], HTML["doc_end"])

    with StringIO() as f:
        read_from(fpath, out=f)
        assert f.getvalue() == expected_out, "Not a valid <p> tag with <strong> tag HTML expression"

def test_cursive_sentence():
    fpath = pkg_resources.resource_filename("tfgpkg", "test/resources/languages/one.hmd")
    text = "this <em>is a cursive</em> test. "
    expected_out = "{}{}{}{}{}".format(HTML["doc_start"], HTML["p_start"], text, HTML["p_end"], HTML["doc_end"])

    with StringIO() as f:
        read_from(fpath, out=f)
        assert f.getvalue() == expected_out, "Not a valid <p> tag with <em> tag HTML expression"

def test_header_sentence():
    fpath = pkg_resources.resource_filename("tfgpkg", "test/resources/languages/two.hmd")
    text = "<h1>Capítulo 1 </h1>"
    expected_out = "{}{}{}".format(HTML["doc_start"], text, HTML["doc_end"])

    with StringIO() as f:
        read_from(fpath, out=f)
        assert f.getvalue() == expected_out, "Not a valid <h1> tag HTML expression"

def test_X_header_sentence():
    fpath = pkg_resources.resource_filename("tfgpkg", "test/resources/languages/three.hmd")
    text = "<h4>Apartado cinco </h4>"
    expected_out = "{}{}{}".format(HTML["doc_start"], text, HTML["doc_end"])

    with StringIO() as f:
        read_from(fpath, out=f)
        assert f.getvalue() == expected_out, "Not a valid <hX> tag HTML expression"

def test_color_sentence():
    fpath = pkg_resources.resource_filename("tfgpkg", "test/resources/languages/four.hmd")
    text = "<span style='color: blue;'>Hello</span> world <span style='color: blue;'>...</span> "
    expected_out = "{}{}{}{}{}".format(HTML["doc_start"], HTML["p_start"], text, HTML["p_end"], HTML["doc_end"])

    with StringIO() as f:
        read_from(fpath, out=f)
        assert f.getvalue() == expected_out, "Not a valid <p> tag with <span style='color'> tag HTML expression"

def test_set_sentence():
    fpath = pkg_resources.resource_filename("tfgpkg", "test/resources/languages/five.hmd")
    text = """<h1>Capítulo 1 </h1>
<h3>Jornalero Español </h3>
<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum </p>""".replace("\n", "")

    expected_out = "{}{}{}".format(HTML["doc_start"], text, HTML["doc_end"])

    with StringIO() as f:
        read_from(fpath, out=f)
        assert f.getvalue() == expected_out, "Not a valid set of HTML expressions with <p> tags"

def test_set_color_sentences():
    fpath = pkg_resources.resource_filename("tfgpkg", "test/resources/languages/six.hmd")
    text = """<h1><span style='background-color: yellow;'>Don</span> <span style='background-color: yellow;'>Quijote</span> </h1>
<h2><span style='color: blue;'>Capítulo</span> <span style='color: blue;'>1</span> </h2>
<p>En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. </p>""".replace("\n", "")

    expected_out = "{}{}{}".format(HTML["doc_start"], text, HTML["doc_end"])

    with StringIO() as f:
        read_from(fpath, out=f)
        assert f.getvalue() == expected_out, "Not a valid set of HTML expressions with <p> tags with <hX>, <span>, <em> and <strong> tags"
