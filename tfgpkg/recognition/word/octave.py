from tensorflow.keras.layers import BatchNormalization, PReLU, Concatenate, UpSampling2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Reshape
from tensorflow.keras.layers import Input, Bidirectional, GRU, Dense

from .base import BaseModel

import os

os.environ["TF_KERAS"] = "1"  # sets Tensorflow as backend for 'keras_octave_conv' Keras layer

from keras_octave_conv import OctaveConv2D
# from .layers import OctConv2D as OctaveConv2D

def ConvLayer(input_layer, filters, kernel, strides, params):
    """Creates a 2D Octave Convolutional stack: Input -> OctaveConv2D -> PReLu -> BatchNormalization -> Output.

    If alpha is 0, only the high channel is computed. If alpha is 1, only the low channel is computed.
    """

    high, low = OctaveConv2D(filters=filters, kernel_size=kernel, strides=strides, **params)(input_layer)
    high, low = BatchNormalization()(high), BatchNormalization()(low)
    
    # Option 1: UpSampling2D + Concat (from the paper)
    upsampled_low = UpSampling2D()(low)
    cat = Concatenate()([high, upsampled_low])

    # Option 2: OctaveConv2D with ratio_out=0.0 (recommended one in OctaveConv2D)
    # cat = OctaveConv2D(filters=..., kernel_size=..., ratio_out=0.0)([high, low])

    return PReLU(shared_axes=[1,2])(cat)

class OctaveModel(BaseModel):
    """Represents a network graph that uses Octave 2D Convolutions"""

    def get_layers(self):
        input_data = Input(shape=self.input_size, name="input")

        params = { "ratio_out": 0.125, "kernel_initializer": "he_uniform" }

        cnn = ConvLayer(input_data, 16, (3,3), (2,2), params)

        cnn = ConvLayer(cnn, 32, (3,3), (1,1), params)

        cnn = ConvLayer(cnn, 40, (2,4), (2,4), params)
        cnn = ConvLayer(cnn, 48, (3,3), (1,1), params)
        # cnn = ConvLayer(cnn, 56, (2,4), (2,4), params)

        cnn = ConvLayer(cnn, 64, (3,3), (1,1), params)

        cnn = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid")(cnn)

        shape = cnn.get_shape()
        nb_units = shape[2] * shape[3]

        blstm = Reshape((shape[1], nb_units))(cnn)

        blstm = Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5))(blstm)
        blstm = Dense(units=nb_units * 2)(blstm)

        blstm = Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5))(blstm)
        output_data = Dense(units=self.model_outputs)(blstm)
        output_data = Activation("softmax", dtype="float32")(output_data)

        return input_data, output_data