import sys
import codecs

from antlr4 import *  # version 4.7.2
from time import time
from languages.htmlColor import HTMLMinidownColorListener
from languages.MinidownColorLexer import MinidownColorLexer
from languages.MinidownColorParser import MinidownColorParser

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
    def __init__(self, listener, fpath=None, text_input=None, output_fpath=None):
        if text_input:
            text = InputStream(text_input)
        elif fpath:
            text = FileStream(fpath, encoding="utf-8")
        else:
            text = InputStream(input("> "))
        
        lexer = MinidownColorLexer(text)
        stream = CommonTokenStream(lexer)
        parser = MinidownColorParser(stream)
        tree = parser.page()

        output_fpath = output_fpath or f"tmp/output_{int(time())}.html"

        with codecs.open(output_fpath, "w", "utf-8") as f:
            html = HTMLMinidownColorListener(f)
            walker = ParseTreeWalker()
            walker.walk(html, tree)

if __name__ == "__main__":
    listener = HTMLMinidownColorListener
    transformer = LanguageTransformer(listener, fpath="examples/two.hmd")
