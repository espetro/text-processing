# Functions to obtain text from word images
from recognition.color import HighlightDetector, ColorExtractor, ColorGroup
from recognition.text import RecognitionNet, FullGatedConv2D
from recognition.vectorizer import StringVectorizer
from recognition.spelling import TextChecker
from recognition.dataunpack import DataUnpack
from recognition.tinydata import TinyData