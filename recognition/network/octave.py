from tensorflow.keras.layers import BatchNormalization, PReLU
from tensorflow.keras.layers import Activation, MaxPooling2D, Reshape
from tensorflow.keras.layers import Input, Bidirectional, GRU, Dense

from tensorflow.python.framework.ops import Tensor
from typing import Tuple

from .base import BaseModel
from .layers import OctConv2D

def ConvLayer(input_layer, filters, kernel, strides, alpha=0):
    """Creates a 2D Octave Convolutional layer.

    If alpha is 0, only the high channel is computed. If alpha is 1, only the
    low channel is computed.
    
    """
    params = { "kernel_initializer": "he_uniform"}

    high, _ = OctConv2D(filters, alpha, kernel, strides, "same" **params)(input_layer)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization()(cnn)
    
    return cnn


class OctaveModel(BaseModel):
    """Represents a network graph that uses Octave 2D Convolutions"""

    def get_layers(self) -> (Tensor, Tensor):
        input_data = Input(name="input", shape=self.input_size)

        cnn = ConvLayer(input_data, 16, (3,3), (2,2), add_dropout=False, add_fullgconv=True)

        cnn = ConvLayer(cnn, 32, (3,3), (1,1), add_dropout=False, add_fullgconv=True)

        cnn = ConvLayer(cnn, 40, (2,4), (2,4), add_dropout=True, add_fullgconv=True)
        cnn = ConvLayer(cnn, 48, (3,3), (1,1), add_dropout=True, add_fullgconv=True)
        cnn = ConvLayer(cnn, 56, (2,4), (2,4), add_dropout=True, add_fullgconv=True)

        cnn = ConvLayer(cnn, 64, (3,3), (1,1), add_dropout=False, add_fullgconv=False)

        cnn = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid")(cnn)

        shape = cnn.get_shape()
        nb_units = shape[2] * shape[3]

        blstm = Reshape((shape[1], nb_units))(cnn)

        blstm = Bidirectional(LSTM(units=nb_units, return_sequences=True, dropout=0.5))(blstm)
        blstm = Dense(units=nb_units * 2)(blstm)

        blstm = Bidirectional(LSTM(units=nb_units, return_sequences=True, dropout=0.5))(blstm)
        output_data = Dense(units=self.model_outputs)(blstm)
        output_data = Activation("softmax", dtype="float32")(output_data)

        return input_data, output_data