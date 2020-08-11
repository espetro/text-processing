from tensorflow.python.framework.ops import Tensor
from typing import Tuple

from .base import BaseModel

class OctaveModel(BaseModel):
    """Represents a network graph that uses Octave 2D Convolutions"""

    @staticmethod
    def get_layers(input_size: Tuple[int, int, int]) -> (Tensor, Tensor):
        pass