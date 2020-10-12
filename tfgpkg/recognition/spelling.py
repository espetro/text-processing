from langdetect import detect, DetectorFactory
from autocorrect import Speller
from typing import List

import pkg_resources

DetectorFactory.seed = 123  # ensure consistent results across multiple runs

class TextChecker:
    """A class that packs functionalities to check the language used in a text and to correct possible misspellings."""
    def __init__(self, predicted_text: List[str]):
        self.lines = predicted_text

        self.checkers = {
            "en": Speller("en"),
            "es": Speller("es")
        }

    def correct(self):
        """Fixes the misspellings in the given text lines."""
        return [self._fix(line) for line in self.lines if len(line) > 0]

    def _fix(self, line):
        try:
            target_lang = detect(line)
        except Exception as e:
            print(e)
            target_lang = "es"

        if target_lang == "es":
            new_line = self.checkers["es"](line)
        else:
            new_line = self.checkers["en"](line)

        return new_line

    @staticmethod
    def add_contextual_information(predicted_text: List[str], lines: List[List], paragraphs: List[int]):
        text_by_lines = []
        for line in lines:
            new_line, num_words = [], len(lines)

            new_line = predicted_text[:num_words]
            predicted_text = predicted_text[num_words:]

            text_by_lines.append(new_line)
            

        return text_by_lines, predicted_text