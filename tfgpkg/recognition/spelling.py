from langdetect import detect, DetectorFactory
from autocorrect import Speller
from typing import List

import pkg_resources

DetectorFactory.seed = 123  # ensure consistent results across multiple runs

class TextChecker:
    """A class that packs functionalities to check the language used in a text and to correct possible misspellings."""
    def __init__(self, lines: List[str]):
        self.lines = lines

        self.checkers = {
            "en": Soeller("en"),
            "es": Soeller("es")
        }

    def correct(self):
        """Fixes the misspellings in the given text lines."""
        return [self._fix(line) for line in self.lines]

    def _fix(self, line):
        target_lang = detect(line)

        if target_lang == "es":
            new_line = self.checkers["es"](line)
        else:
            new_line = self.checkers["en"](line)

        return new_line