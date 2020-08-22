from typing import List
from langdetect import detect, DetectorFactory

try:
    # jamspell requires a specific installation (along with C++'s Swig 3.0)
    from jamspell import TSpellCorrector
except ModuleNotFoundError:
    from spellchecker import SpellChecker

import importlib_resources as pkg_resources  # backport of core 3.7 library

DetectorFactory.seed = 123  # ensure consistent results across multiple runs

class TextChecker:
    """A class that packs functionalities to check the language used in a text and to correct possible misspellings.

    Source:
        https://github.com/Mimino666/langdetect.git
        https://github.com/barrust/pyspellchecker.git (windows build)
        https://github.com/bakwc/JamSpell.git (linux / mac build)
    """
    MODEL_PATHS = {
        "en": str(pkg_resources.files("recognition.data").joinpath("en.bin")),
        "es": str(pkg_resources.files("recognition.data").joinpath("es.bin")),
        "fr": str(pkg_resources.files("recognition.data").joinpath("fr.bin"))
    }

    def __init__(self, lines: List[str]):
        self.lines = lines
        self.output = []

        # self.checkers = { "en": SpellChecker(distance=2, language="en"), "es": SpellChecker(distance=2, language="es"), "fr": SpellChecker(distance=2, language="fr") }
        self.checkers = {
            "en": TSpellCorrector(),
            "es": TSpellCorrector(),
            "fr": TSpellCorrector()
        }

        for nm, speller in self.checkers.items():
            speller.LoadLangModel(TextChecker.MODEL_PATHS.get(nm))  # loads en.bin, fr.bin, es.bin

    def correct(self):
        """Fixes the misspellings in the given text lines."""
        for line in lines:
            target_lang = detect(line)
            new_line = self.checkers[target_lang].FixFragment(line)
            self.output.append(new_line)
        
        return self.output