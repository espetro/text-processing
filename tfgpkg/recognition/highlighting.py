from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import SeparableConv2D
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from ..preproc.word import Quantize

import matplotlib.pyplot as plt
import skimage.exposure as exp
import pkg_resources
import numpy as np
import cv2

class HighlightDetector:
    """A class"""
    
    NUM_CLASSES = 2
    TARGET_SIZE = (150,150, 3)
    MODEL_PATH = pkg_resources.resource_filename("tfgpkg.recognition.data", "highlight_model_mini_45e_64bz_weights.ckpt")

    def __init__(self, target_size=None, epochs=1):
        self.input_size = target_size or HighlightDetector.TARGET_SIZE

        self.net: KerasClassifier = KerasClassifier(
            build_fn=HighlightDetector._build_model,
            input_size=self.input_size,
            epochs=1,
            batch_size=32,
            verbose=0
        )

    def train(self, X_train, Y_train, epochs=30, plot=False):
        training_results = self.net.fit(X_train, Y_train, verbose=1)
        if plot:
            HighlightDetector.plot_results(training_results)

    def predict(self, X_test):
        """"""
        return self.net.model.predict_classes(X_test).flatten()

    def cross_validate(self, X, Y, k=10, epochs=30, batch_sz=128):
        kfold = StratifiedKFold(n_splits=k, shuffle=True)
        results = cross_val_score(self.net, X, Y, cv=kfold)

        print(f"Baseline: {(results.mean() * 100):.2f}% ({(results.std()*100):.2f}%)")

    def load_model(self, fpath=None):
        fpath = fpath or str(HighlightDetector.MODEL_PATH)

        # initialize the network
        dummyX, dummyY = np.zeros((1,150,150,3)), np.zeros((1))
        _ = self.net.fit(dummyX, dummyY, verbose=0)

        self.net.model.load_weights(fpath)

    @staticmethod    
    def minmax_scaler(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    @staticmethod
    def preprocess(image, brightness=20, contrast_gain=0.05):
        """
        Parameters
        ----------
            image: ndarray image in RGB mode
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:,:,2] += brightness
        
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        image = exp.adjust_sigmoid(image, cutoff=0.5 - contrast_gain)
        
        image = Quantize.reduce_palette(image, num_colors=4)
        return HighlightDetector.minmax_scaler(image)

    @staticmethod
    def decode(prediction):
        return {1: "Non-highlighted", 0: "Highlighted"}.get(prediction)
        
    @staticmethod
    def _build_model(input_size):
        model = Sequential()
    
        model.add(SeparableConv2D(32, (3,3), activation="relu", input_shape=(150, 150, 3), name="conv1"))
        model.add(MaxPooling2D((2,2), name="pool1"))
        
        model.add(SeparableConv2D(16, (3,3), activation="relu", name="conv2"))
        model.add(MaxPooling2D((2,2), name="pool2"))

        model.add(Conv2D(8, (3,3), activation="relu", name="conv3"))
        model.add(MaxPooling2D((2,2), name="pool3"))

        model.add(Conv2D(4, (3,3), activation="relu", name="conv4"))
        model.add(MaxPooling2D((2,2), name="pool4"))
        
        model.add(Flatten())
        
        model.add(Dense(8, activation='relu', name="dense1"))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid', name="dense2"))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def plot_results(training_results):
        _, axes = plt.subplots(1,2, figsize=(10,5))
        axes = axes.flatten()

        axes[0].plot(training_results.history["accuracy"])
        axes[0].set_title("Training Accuracy")

        axes[1].plot(training_results.history["loss"])
        axes[1].set_title("Training Loss")

        plt.show()
