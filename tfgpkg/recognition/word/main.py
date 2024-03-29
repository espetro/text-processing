from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ..dataunpack import DataUnpack
from .base import BaseModel
from .depthwise import DepthwiseModel
from .octave import OctaveModel
from .gated import GatedModel
from .baseline import BaselineModel

from .metrics import cer, wer

from os.path import expanduser
from typing import List, Tuple
from enum import Enum

import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf
import editdistance
import numpy as np
import sys
import os

if sys.version_info.minor < 7:
    import importlib_resources as pkg_resources
elif sys.version_info.minor >= 7:
    import pkg_resources


class Arch(Enum):
    """
    Octave: Applies octave convolutions
    Gated: Applies full gated convolutions
    Depthwise: Applies depthwise convolutions  
    """
    Baseline = 0
    Octave = 1
    Depthwise = 2
    Gated = 3
    Baseline = 4

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

        EarlyStopping(monitor=monitor, min_delta=1e-4, patience=5,
            restore_best_weights=True, verbose=verbose),

        ReduceLROnPlateau(monitor=monitor, min_delta=1e-8, factor=0.2,
            patience=10, verbose=verbose)
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
    INPUT_SIZE = (266, 64, 1)
    
    # pretrained model
    MODEL_PATH = pkg_resources.resource_filename("tfgpkg.recognition.data", "crnn_model_1e_weights.ckpt")

    ASCII_CHAR = " !\"#$%&'()*+,-.0123456789:;<>@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    LATIN_CHAR = " !\"#$%&'()*+,-.0123456789:;<>@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáÁéÉíÍóÓúÚëËïÏüÜñÑçÇâÂêÊîÎôÔûÛàÀèÈùÙ"
    
    DECODER_CONFIG = { "greedy": False, "beam_width": 5, "top_paths": 1 }

    def __init__(self, logdir=None, input_size=None, arch=Arch.Gated,
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

    def load_weights(self, fpath: str = None):
        """ Load model weights from a checkpoint file."""
        fpath = fpath or str(RecognitionNet.TRAINED_WEIGHTS_PATH)
        self.model.load_weights(fpath)
    
    @staticmethod
    def load_model(fpath: str = None, input_size: Tuple[int, int] = (192, 48)):
        """ Load a model from a checkpoint file."""
        fpath = fpath or str(RecognitionNet.TRAINED_MODEL_PATH)
        
        net = RecognitionNet(".", input_size)
        net.model = tf.keras.models.load_model(fpath, custom_objects={ "ctc_loss_lambda_func": BaseModel.ctc_loss_lambda_func })
        return net
           
    @staticmethod
    def compute_wer(true_label: List[str], pred_label: List[str]):
        """Computes the WER based on the Levenshtein distance"""
        # return wer(true_labels.join(" "), pred_labels.join(" "))
        num_true_strs, num_preds_strs = len(true_label) + 1, len(pred_label) + 1

        mat = np.zeros((num_true_strs) * (num_preds_strs), dtype=np.uint8)
        mat = mat.reshape((num_true_strs, num_preds_strs))
        for i in range(num_true_strs):
            for j in range(num_preds_strs):
                if i == 0:
                    mat[0, j] = j
                elif j == 0:
                    mat[i, 0] = i

        # computation
        for i in range(1, num_true_strs):
            for j in range(1, num_preds_strs):
                if true_label[i - 1] == pred_label[j - 1]:
                    mat[i, j] = mat[i - 1, j - 1]
                else:
                    substitution = mat[i - 1, j - 1] + 1
                    insertion    = mat[i, j - 1] + 1
                    deletion     = mat[i - 1, j] + 1
                    mat[i, j] = min(substitution, insertion, deletion)

        return mat[len(true_label), len(pred_label)]

    @staticmethod
    def compute_cer(true_label: str, pred_label: str):
        """Computes the CER based on Levenshtein distance"""
        # return np.mean([cer(tru, prd) for tru, prd in zip(true_labels, pred_labels)])
        dist = editdistance.eval(true_label, pred_label)
        word_length = max(len(true_label), len(pred_label))
        return (dist / word_length)

    def _build_model(self, optimizer, arch: Arch):
        """Configures the HTR Model for training/predict.

        Parameters
        ----------
            arch: Arch          
        """
        if arch is None:
            raise ValueError(f"Wrong architecture value {arch} (only Arch. Octave, Depthwise or Gated are available)")
        elif arch is Arch.Baseline:
            return BaselineModel(self.input_size, self.model_outputs, optimizer).get_model()
        elif arch is Arch.Gated:
            return GatedModel(self.input_size, self.model_outputs, optimizer).get_model()
        elif arch is Arch.Octave:
            return OctaveModel(self.input_size, self.model_outputs, optimizer).get_model()
        elif arch is Arch.Depthwise:
            return DepthwiseModel(self.input_size, self.model_outputs, optimizer).get_model()

    def predict(self, x, batch_size=None, verbose=0, steps=1, callbacks=None, max_queue_size=10, workers=1,
        use_multiprocessing=False, ctc_decode=True):
        """
        Model predicting on data yielded (predict function has support to generator).
        A predict() abstration function of TensorFlow 2.

        Provide x parameter of the form: yielding [x].

        :param: See tensorflow.keras.Model.predict()
        :return: raw data on `ctc_decode=False` or CTC decode on `ctc_decode=True` (both with probabilities)
        """
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
    def preprocess(image, aspect_ratio: float, target_size: Tuple[int, int]):
        """Preprocessing function for an image (integer np.array) in RGB mode.

        Source:
            recognition.DataUnpack.unpack_set method
        """
        if target_size and aspect_ratio:
            image = DataUnpack.resize("", image, target_size, aspect_ratio)
        
        image = image.transpose()
        image = np.expand_dims(image, axis=-1)  # add a sigle, grayscale channel to the image
        image = image / 255.  # normalize it
        return np.asarray(image)


# ===============================================


if __name__ == "__main__":
    net = RecognitionNet(".", input_size=(192, 48, 1))
    net.load_model()
    net.summary()
