from tensorflow.keras.layers import BatchNormalization, PReLU, SeparableConv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Reshape
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense

from tensorflow.python.framework.ops import Tensor
from typing import Tuple

from .base import BaseModel

def SepConvStack(input_layer, filters, kernel, strides):
    """Retrieves the CNN stack: Input -> SeparableConv2D -> PReLu -> BatchNorm -> Output"""
    params = { "padding": "same", "kernel_initializer": "he_uniform"}

    cnn = SeparableConv2D(filters, kernel, strides, **params)(input_layer)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization()(cnn)
    
    return cnn

class DepthwiseModel(BaseModel):
    """Represents a network graph that uses Depthwise 2D Convolutions."""

    def get_layers(self) -> (Tensor, Tensor):
        input_data = Input(name="input", shape=self.input_size)

        cnn = SepConvStack(input_data, 16, (3,3), (2,2))
        cnn = SepConvStack(cnn, 32, (3,3), (1,1))

        cnn = SepConvStack(cnn, 40, (2,4), (2,4))
        cnn = SepConvStack(cnn, 48, (3,3), (1,1))
        cnn = SepConvStack(cnn, 56, (2,4), (2,4))

        cnn = SepConvStack(cnn, 64, (3,3), (1,1))

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