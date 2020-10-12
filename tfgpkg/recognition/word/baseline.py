from tensorflow.keras.layers import BatchNormalization, PReLU
from tensorflow.keras.layers import Input, Activation, MaxPooling2D, Reshape
from tensorflow.keras.layers import Conv2D, Bidirectional, GRU, Dense

from tensorflow.python.framework.ops import Tensor
from typing import Tuple

from .base import BaseModel

def ConvLayer(input_layer, filters, kernels, strides):
    opts = dict(padding="same", kernel_initializer="he_uniform")

    cnn = Conv2D(filters=filters, kernel_size=kernels[0], strides=strides, **opts)(input_layer)    
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization()(cnn)

    return cnn

class BaselineModel(BaseModel):
    """Represents a network graph that uses Common 2D Convolutions with GRU recurrent units."""

    def get_layers(self) -> (Tensor, Tensor):
        input_data = Input(name="input", shape=self.input_size)

        cnn = ConvLayer(input_data, 16, [(3,3), (3,3)], (2,2))

        cnn = ConvLayer(cnn, 32, [(3,3), (3,3)], (1,1))

        cnn = ConvLayer(cnn, 40, [(2,4), (3,3)], (2,4))
        cnn = ConvLayer(cnn, 48, [(3,3), (3,3)], (1,1))
        cnn = ConvLayer(cnn, 56, [(2,4), (3,3)], (2,4))

        cnn = ConvLayer(cnn, 64, [(3,3), (None, None)], (1,1))

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