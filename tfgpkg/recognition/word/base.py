from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import Tensor
from typing import Tuple
from numpy import ndarray

import tensorflow as tf
import tensorflow.keras.backend as K

# ===============================

class BaseModel:
    """Represents a Keras network graph.
    
    If a new model needs to be implemented, it can use BaseModel as parent class,
    overriding the following methods:
        - get_layers
    
    If the new model uses a different default optimizer, just pass it:
        super().__init__(optimizer=my_new_default_optimizer)
    
    You can also override the __init__ method.
    """

    def __init__(self, input_size: Tuple[int, int, int], model_outputs, optimizer=None):
        self.input_size = input_size
        self.model_outputs = model_outputs
        self.optimizer = optimizer or Adam()

    def get_model(self):
        """Returns the network's Keras model"""
        _in, _out = self.get_layers()
 
        model = Model(inputs=_in, outputs=_out)
        model.compile(optimizer=self.optimizer, loss=BaseModel.ctc_loss_lambda_func)

        return model

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
        pass