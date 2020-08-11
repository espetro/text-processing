from tensorflow.python.framework.ops import Tensor
from typing import Tuple

from .base import BaseModel

class DepthwiseModel(BaseModel):
    """Represents a network graph that uses Depthwise 2D Convolutions."""

    @staticmethod
    def get_layers(input_size: Tuple[int, int, int]) -> (Tensor, Tensor):
        """Builds the network graph and returns its input and output layers"""
        pass