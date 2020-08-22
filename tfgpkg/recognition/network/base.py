from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Reshape
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

from tensorflow.python.framework.ops import Tensor
from typing import Tuple
from numpy import ndarray

def ConvLayer(input_layer, filters, kernels, strides, add_dropout=False, add_fullgconv=False, dtype="float32"):
    cnn = Conv2D(filters=filters, kernel_size=kernels[0], strides=strides, padding="same", kernel_initializer="he_uniform")
        (input_layer)
        
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization()(cnn)
    return cnn

# ===============================

class BaseModel:
    """Represents a Keras network graph with common 2D convolutions.
    
    If a new model needs to be implemented, it can use BaseModel as parent class,
    overriding the following methods:
        - get_layers
    
    If the new model uses a different default optimizer, just pass it:
        super().__init__(optimizer=my_new_default_optimizer)
    
    You can also override the __init__ method.
    """

    def __init__(self, input_size: Tuple[int, int, int], optimizer=None):
        self.input_size = input_size

        _in, _out = self.get_layers()

        optimizer = optimizer or RMSprop(learning_rate=5e-4)
        model = Model(inputs=_in, outputs=_out)

        model.compile(optimizer=optimizer, loss=BaseModel.ctc_loss_lambda_func)
        return model

    def get_model(self):
        """Returns the network's Keras model"""
        return self.model

    @staticmethod
    def ctc_loss_lambda_func(y_true: ndarray, y_pred: ndarray) -> float:
        """Function for computing the CTC loss"""
        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)

        # y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
        # output of every model is softmax
        # so sum across alphabet_size_1_hot_encoded give 1
        #               string_length give string length
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

        # y_true strings are padded with 0
        # so sum of non-zero gives number of characters in this string
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        # average loss across all entries in the batch
        loss = tf.reduce_mean(loss)

        return loss

    def get_layers(self) -> (Tensor, Tensor):
        """Builds the network graph and returns its input and output layers"""
        input_data = Input(name="input", shape=self.input_size)

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