# Generated from Minidown.g4 by ANTLR 4.7.2
import sys
import re

from antlr4 import *
from languages.MinidownColorParser import MinidownColorParser
from languages.MinidownColorListener import MinidownColorListener

class HTMLMinidownColorListener(MinidownColorListener):
    """
    Listener methods to translate custom code to HTML5 code

    Parameters
    ----------
        output: File object
            A writable file object
    """
    def __init__(self, output):
        self.output = output
        self.output.write(u'\ufeff')  # encodes the file as UTF-8

    def enterPage(self, ctx:MinidownColorParser.PageContext):
        self.output.write("<html>\n\r<head><title>Result</title><meta charset='UTF-8'/></head>\n<body>")
        pass

    def exitPage(self, ctx:MinidownColorParser.PageContext):
        self.output.write("\n</body>\n</html>\n")
        pass

    def enterHeader(self, ctx:MinidownColorParser.HeaderContext):
        levels = len(ctx.level.text)
        self.output.write(f"\n<h{levels}>\n\t")

    def exitHeader(self, ctx:MinidownColorParser.HeaderContext):
        """
        Parameters
        ----------
            ctx, MinidownParser.HeaderContext
                Context with atrributes
                    level: CommonToken
                        Use ctx.level.text or ctx.HEADER().getText() to get str
                    value: TextContext
                        Use ctx.value.getText() or ctx.text().getText() to get str
        """
        levels = len(ctx.level.text)
        self.output.write(f"\n</h{levels}>")

    def enterParagraph(self, ctx:MinidownColorParser.ParagraphContext):
        self.output.write("\n<p>\n\t")

    def exitParagraph(self, ctx:MinidownColorParser.ParagraphContext):
        self.output.write("\n</p>")

    def exitWord(self, ctx:MinidownColorParser.WordContext):       
        word = ctx.value.text
        prefix_style, suffix_style = HTMLMinidownColorListener.get_styles(ctx)
        colors_tag1, colors_tag2 = HTMLMinidownColorListener.get_colors(ctx)

        self.output.write(f"{prefix_style}{colors_tag1}{word}{colors_tag2}{suffix_style} ")

    @staticmethod
    def get_styles(ctx:MinidownColorParser.WordContext):
        pref, suf = "", ""
        if ctx.prefix:
            pref = {
                "@": "<em>",
                "$": "<strong>"
            }.get(ctx.prefix.text)
        if ctx.suffix:
            suf = {
                "@": "</em>",
                "$": "</strong>"
            }.get(ctx.suffix.text)

        return pref, suf

    @staticmethod
    def get_colors(ctx:MinidownColorParser.WordContext):
        font, bg = ctx.font.text, ctx.bg.text
        
        # by default, CSS text color is BLACK
        if font != "black" and bg != "None":  
            result = (f"<span style='color: {font}; background-color: {bg}'>", "</span>")
        elif font == "black" and bg != "None":
            result = (f"<span style='background-color: {bg};'>", "</span>")
        elif font != "black" and bg == "None":
            result = (f"<span style='color: {font};'>", "</span>")
        else:
            result = ("", "")

        return result

