from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ..dataunpack import DataUnpack
from .base import BaseModel
from .depthwise import DepthwiseModel
from .octave import OctaveModel
from .gated import GatedModel

from os.path import expanduser
from enum import Enum

import os
import numpy as np
import editdistance
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import importlib_resources as pkg_resources

class Arch(Enum):
    """
    Base: Applies common convolutions
    Octave: Applies octave convolutions
    Gated: Applies full gated convolutions
    Depthwise: Applies depthwise convolutions  
    """
    Base = 1
    Octave = 2
    Depthwise = 3
    Gated = 4

def set_callbacks(logdir, verbose=0, monitor="val_loss"):
    """Setup a list of Tensorflow Keras callbacks"""
    log_file = {
        "csv": os.path.join(logdir, "epochs.log"),
        "ckpt": os.path.join(logdir, "checkpoint.params")
    }

    callbacks = [
        CSVLogger(filename=log_file["csv"], separator=";", append=True),

        # TensorBoard(log_dir=logdir, histogram_freq=20, profile_batch=0,
        #     write_graph=True, write_images=False, update_freq="epoch"),

        ModelCheckpoint(filepath=log_file["ckpt"], monitor=monitor,
            save_best_only=True, save_weights_only=True, verbose=verbose),

        EarlyStopping(monitor=monitor, min_delta=1e-8, patience=20,
            restore_best_weights=True, verbose=verbose),

        ReduceLROnPlateau(monitor=monitor, min_delta=1e-8, factor=0.2,
            patience=15, verbose=verbose)
    ]

    return callbacks

# =========================================

class RecognitionNet:
    """Class representing the neural network used for recognizing word images
    
    Default values:
      + INPUT_SIZE = (266, 64, 1)
      + MAX_WORD_LENGTH = 64
      + CHARSET = RecognitionNet.LATIN_CHAR

    Parameters
    ----------
        logdir: str
        input_size: Tuple[int, int, int]
        arch: Arch
        shrink: bool
            If True, the model is quantized and pruned
        charset: str
        optimizer: ABCMeta
            A Tensorflow Keras optimizer
        decoder_conf: Dict
        verbose: int
    """
    
    # pretrained model
    MODEL_PATH = "" # pkg_resources.files("recognition.data").joinpath("text_weights/crnn_model_1e_weights.ckpt")

    ASCII_CHAR = " !\"#$%&'()*+,-.0123456789:;<>@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    LATIN_CHAR = " !\"#$%&'()*+,-.0123456789:;<>@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáÁéÉíÍóÓúÚëËïÏüÜñÑçÇâÂêÊîÎôÔûÛàÀèÈùÙ"
    
    DECODER_CONFIG = { "greedy": False, "beam_width": 5, "top_paths": 1 }

    def __init__(self, logdir=None, input_size=None, arch=Arch.Base,
        shrink=False, charset=None, optimizer=None, decoder_conf=None, verbose=0):

        if input_size is None:
            raise ValueError("An input shape (h, w, ch) is needed")

        self.input_size = input_size 
        self.charset = charset or RecognitionNet.LATIN_CHAR
        self.model_outputs = len(self.charset) + 1
        
        if logdir:
            self.logdir = logdir
        else:
            self.logdir = f"{expanduser('~')}{os.sep}.logs"

        self.callbacks = set_callbacks(self.logdir, verbose, monitor="val_loss")
        self.decoder_conf = decoder_conf or RecognitionNet.DECODER_CONFIG

        self.model = self._build_model(optimizer, arch)

    def load_chkpt(self, fpath: str=None):
        """ Load a model with checkpoint file."""
        fpath = fpath or str(RecognitionNet.MODEL_PATH)
        self.model.load_weights(fpath)
           
    @staticmethod
    def compute_wer(true_labels, pred_labels):
        wer = [not np.array_equaldist(tru, prd) for tru, prd in zip(true_labels, pred_labels)]
        return np.mean(wer)

    @staticmethod
    def compute_cer(true_labels, pred_labels):
        cer = []
        for tru, prd in zip(true_labels, pred_labels):
            dist = editdistance.eval(tru, prd)
            word_length = max(len(tru), len(prd))
            cer.append(dist / word_length)
        
        return np.mean(cer)

    def _build_model(self, optimizer=None, arch: Arch=None):
        """Configures the HTR Model for training/predict.

        Parameters
        ----------
            arch: Arch          
        """
        if arch is Arch.Base or arch is None:
            return BaseModel(self.input_size, optimizer).get_model()
        elif arch is Arch.Gated:
            return GatedModel(self.input_size, optimizer).get_model()
        elif arch is Arch.Octave:
            return OctaveModel(self.input_size, optimizer).get_model()
        elif arch is Arch.Depthwise:
            return DepthwiseModel(self.input_size, optimizer).get_model()

    def predict(self, x, batch_size=None, verbose=0, steps=1, callbacks=None, max_queue_size=10, workers=1,
        use_multiprocessing=False, ctc_decode=True):
        """
        Model predicting on data yielded (predict function has support to generator).
        A predict() abstration function of TensorFlow 2.

        Provide x parameter of the form: yielding [x].

        :param: See tensorflow.keras.Model.predict()
        :return: raw data on `ctc_decode=False` or CTC decode on `ctc_decode=True` (both with probabilities)
        """

        self.model._make_predict_function()

        if verbose == 1:
            print("Model Predict")

        out = self.model.predict(
            x=x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size,
            workers=workers, use_multiprocessing=use_multiprocessing
        )

        if not ctc_decode:
            return np.log(out.clip(min=1e-8))

        steps_done = 0
        if verbose == 1:
            print("CTC Decode")
            progbar = tf.keras.utils.Progbar(target=steps)

        batch_size = int(np.ceil(len(out) / steps))
        input_length = len(max(out, key=len))

        predicts, probabilities = [], []
        dconf1 = self.decoder_conf["greedy"]
        dconf2 = self.decoder_conf["beam_width"]
        dconf3 = self.decoder_conf["top_paths"]

        while steps_done < steps:
            index = steps_done * batch_size
            until = index + batch_size

            x_test = np.asarray(out[index:until])
            x_test_len = np.asarray([input_length for _ in range(len(x_test))])

            decode, log = K.ctc_decode(
                x_test, x_test_len, greedy=dconf1, beam_width=dconf2, top_paths=dconf3
            )

            probabilities.extend([np.exp(x) for x in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts.extend(np.swapaxes(decode, 0, 1))

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

        return (predicts, probabilities)

    @staticmethod
    def preprocess(image, aspect_ratio=None, target_size=None):
        """Preprocessing function for an image (integer np.array) in RGB mode.

        Source:
            recognition.DataUnpack.unpack_set method
        """
        target_size = target_size or RecognitionNet.INPUT_SIZE  # defaults to (256, 64, 1)

        if target_size and aspect_ratio:
            image = DataUnpack.resize("", image, target_size, aspect_ratio)
        
        image = image.transpose()
        image = np.expand_dims(image, axis=-1)  # add a sigle, grayscale channel to the image
        image = image / 255.  # normalize it
        return np.asarray(image)


# ===============================================


if __name__ == "__main__":
    net = RecognitionNet(".")
    net.load_chkpt()
    net.summary()