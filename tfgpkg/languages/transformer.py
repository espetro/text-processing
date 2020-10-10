from antlr4 import InputStream, FileStream, CommonTokenStream, ParseTreeWalker
from antlr4.error.ErrorListener import ErrorListener
from tempfile import NamedTemporaryFile
from typing import List, Tuple
from time import time

from tfgpkg.languages.MinidownColorParser import MinidownColorParser
from tfgpkg.languages.MinidownColorLexer import MinidownColorLexer
from tfgpkg.languages.htmlColor import HTMLMinidownColorListener

import codecs
import sys


class ExceptionListener(ErrorListener):
    """Reroutes all ANTLR4 errors through the Python exception system"""
    INSTANCE = None

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        print(e, file=sys.stderr)
        raise SyntaxError(f"Invalid input when parsing line {line} column {column}: {msg}\n")


class LanguageTransformer:
    """
    Transforms word-color format to a given output language.

    Parameters
    ----------
        listener: class object 
            A listener object, that inherits from MinidownColorLexer and antlr4.Lexer

        fpath: str, default None
            The filepath of the input file. If None, then the sys.stdin is used.

        output_fpath: str, default None
            The filepath of the output file. If None, a random name is given to the file.

    """
    def __init__(self, listener, fpath=None, text_input=None, output_fpath=None, output_file=None):
        if text_input is not None:
            text = InputStream(text_input)
        elif fpath is not None:
            text = FileStream(fpath, encoding="utf-8")
        else:
            text = ""
            # raise ValueError("Expected either a file input or a string input")
        
        lexer = MinidownColorLexer(text)
        stream = CommonTokenStream(lexer)

        parser = MinidownColorParser(stream)
        parser.removeErrorListeners()
        parser.addErrorListener(ExceptionListener())

        tree = parser.page()

        if output_fpath is not None:
            f = codecs.open(output_fpath, "w", "utf-8")
        elif output_file is not None:
            f = output_file
        else:
            f = NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=f"{int(time())}")

        html = HTMLMinidownColorListener(f)
        walker = ParseTreeWalker()
        walker.walk(html, tree)

        if output_file is None:
            f.close()


    @staticmethod
    def collect_word_color(color_input: List[Tuple[str, str, str, bool]], text_input: List[str]) -> str:
        """
        Collect the text input and color input and build a list of tuples (word, font color, bg color).
        
        The text input consists of a list of word predictions, and the color input consists of a list of tuples
        (color1, color2, ..., class), with the most predominant K colors and a class label (highlighted or not).
        Usually, the most predominant color belongs to the background. If the word is not highlighted, then the 2nd most
        predominant color belongs to the font, otherwise it is the 3rd.

        This method outputs a string where each resulting tuple is formatted as:
        '(' WORD ' , ' FONT_COLOR ' , ' BG_COLOR ')'
        
        Please note the whitespaces.
        """
        if len(color_input) != len(text_input):
            raise ValueError("Expected both the text and color input lists to have the same length.")

        result = ""
        for (color1, color2, color3, is_highlighted), word in zip(text_input, color_input):
            bg_color, font_color = color1, color2

            if is_highlighted:
                bg_color, font_color = color1, color3

            result += f"({word} , {font_color} , {bg_color}) "

        return result.strip()  # remope the trailing whitespaces


if __name__ == "__main__":
    listener = HTMLMinidownColorListener
    transformer = LanguageTransformer(listener, fpath="examples/two.hmd")
