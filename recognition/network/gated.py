from tensorflow.keras.layers import Dropout, BatchNormalization, PReLU
from tensorflow.keras.layers import Input, Activation, MaxPooling2D, Reshape
from tensorflow.keras.layers import Conv2D, Bidirectional, GRU, Dense
from tensorflow.keras.constraints import MaxNorm

from tensorflow.python.framework.ops import Tensor
from typing import Tuple

from .base import BaseModel
from .layers import FullGatedConv2D

def ConvLayer(input_layer, filters, kernels, strides, add_dropout=False, add_fullgconv=False, dtype="float32"):
    cnn = Conv2D(filters=filters, kernel_size=kernels[0], strides=strides, padding="same", kernel_initializer="he_uniform")
        (input_layer)
        
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization()(cnn)

    if add_fullgconv:
        cnn = FullGatedConv2D(filters=filters, kernel_size=kernels[-1], padding="same",
            kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)

        if add_dropout:  # only add it when adding a FullGatedConv layer previously
            cnn = Dropout(rate=0.2)(cnn)

    return cnn

class GatedModel(BaseModel):
    """Represents a network graph that uses Full Gated 2D Convolutions."""

    @staticmethod
    def get_layers(input_size: Tuple[int, int, int]) -> (Tensor, Tensor):
        """Builds the network graph and returns its input and output layers"""
        input_data = Input(name="input", shape=input_size)

        cnn = ConvLayer(input_data, 16, [(3,3), (3,3)], (2,2), add_dropout=False, add_fullgconv=True)

        cnn = ConvLayer(cnn, 32, [(3,3), (3,3)], (1,1), add_dropout=False, add_fullgconv=True)

        cnn = ConvLayer(cnn, 40, [(2,4), (3,3)], (2,4), add_dropout=True, add_fullgconv=True)
        cnn = ConvLayer(cnn, 48, [(3,3), (3,3)], (1,1), add_dropout=True, add_fullgconv=True)
        cnn = ConvLayer(cnn, 56, [(2,4), (3,3)], (2,4), add_dropout=True, add_fullgconv=True)

        cnn = ConvLayer(cnn, 64, [(3,3), (None, None)], (1,1), add_dropout=False, add_fullgconv=False)

        cnn = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid")(cnn)

        shape = cnn.get_shape()
        nb_units = shape[2] * shape[3]

        bgru = Reshape((shape[1], nb_units))(cnn)

        bgru = Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5))(bgru)
        bgru = Dense(units=nb_units * 2)(bgru)

        bgru = Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5))(bgru)
        output_data = Dense(units=self.model_outputs)(bgru)
        output_data = Activation("softmax", dtype="float32")(output_data)

        return input_data, output_data