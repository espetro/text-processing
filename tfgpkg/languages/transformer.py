import sys
import codecs

from antlr4 import InputStream, FileStream, CommonTokenStream, ParseTreeWalker
from tfgpkg.languages.htmlColor import HTMLMinidownColorListener
from tfgpkg.languages.MinidownColorLexer import MinidownColorLexer
from tfgpkg.languages.MinidownColorParser import MinidownColorParser
from antlr4.error.ErrorListener import ErrorListener
from tempfile import NamedTemporaryFile
from time import time


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

if __name__ == "__main__":
    listener = HTMLMinidownColorListener
    transformer = LanguageTransformer(listener, fpath="examples/two.hmd")
