from preproc import Quantize
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras.layers import SeparableConv2D
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from recognition.colorthief import ColorThief
from PIL import Image

import importlib_resources as pkg_resources  # backport of core 3.7 library
import matplotlib.pyplot as plt
import skimage.exposure as exp
import pickle as pk
import pandas as pd
import numpy as np
import numbers
import keras
import cv2
import os


class HighlightDetector:
    """A class"""
    
    NUM_CLASSES = 2
    TARGET_SIZE = (150,150, 3)
    MODEL_PATH = pkg_resources.files("recognition.data").joinpath("highlight_model_mini_45e_64bz_weights.ckpt")
    # remember to add __init__.py to data/ folder

    def __init__(self, target_size=None, epochs=1):
        self.input_size = target_size or HighlightDetector.TARGET_SIZE

        self.net = KerasClassifier(
            build_fn=HighlightDetector._build_model,
            input_size=self.input_size,
            epochs=1,
            batch_size=32,
            verbose=0
        )

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

    def load_weights(self, fpath=None):
        fpath = fpath or str(HighlightDetector.MODEL_PATH)

        # initialize the network
        dummyX, dummyY = np.zeros((1,150,150,3)), np.zeros((1))
        _ = self.net.fit(dummyX, dummyY, verbose=0)

        self.net.model.load_weights(fpath)


# =====================================


class ColorExtractor:
    """A class representing the color extraction algorithm.
    Given an image, it obtains a set of color names.
    It uses a custom clustering algorithm under the hood (namely MMCQ).

    Source:
        https://github.com/fengsp/color-thief-py.git
        http://www.leptonica.com

    Parameters
    ----------
        image: ndarray of shape (X,Y) in RGB mode
    """
    def __init__(self, image):
        if not issubclass(image.dtype.type, numbers.Integral):
            self.image = Image.fromarray((image * 255.).astype(np.uint8))

        self.image = Image.fromarray(image)
        
    def palette(self, num_colors=3, precise=True, preprocess=True):
        quality = {True: 1, False: 8}.get(precise)
        extractor = ColorThief(self.image)
        
        colors = extractor.get_palette(color_count=num_colors, quality=quality)
        if preprocess:
            colors = ColorGroup.preprocess(colors)
        
        return colors[:num_colors]

# =====================================

def expand_colors(color_names_file):
    """Extends the KNN classifier by adding new colors to the labels / samples arrays.

    Source:
        https://github.com/algolia/color-extractor (color_names.npz)
    """

    rng, new_labels, new_samples = range(240, 256, 1), [], []
    for c1 in rng:
        for c2 in rng:
            for c3 in rng:
                new_samples.append((c1,c2,c3))
    new_labels = ["white"] * len(new_samples)

    with open(color_names_file, "rb") as f:
        old_colors = np.load(color_names_file, allow_pickle=True)

    samples, labels = old_colors.get("samples"), old_colors.get("labels")
    samples = np.concatenate((samples, np.array(new_samples)))
    labels = np.concatenate((labels, np.array(new_labels)))

    model = KNeighborsClassifier(n_neighbors, weights=weights)
    model.fit(samples, labels)

    np.savez("color_names_white.npz", samples=samples, labels=labels)
    with open("knn_10_uniform_white.pk", "wb") as f:
        pk.dump(model, f)


class ColorGroup:
    """
    A class representing the color naming algorithm.
    Given an RGB tuple, it returns the name of the closest CSS3 name color.
    It uses K-NN under the hood.

    Source:
        https://blog.algolia.com/how-we-handled-color-identification/
        https://blog.xkcd.com/2010/05/03/color-survey-results/
    """
    # KNearestNeighbor with K=10 and uniform weights
    MODEL_PATH = pkg_resources.files("recognition.data").joinpath("knn_10_uniform_white_mini.pk")
    LABELS_PATH = pkg_resources.files("recognition.data").joinpath("color_names_white_mini.npz")

    def __init__(self):
        """Loads the saved model"""
        with open(str(ColorGroup.MODEL_PATH), "rb") as f:
            self.model = pk.load(f)

    def predict(self, color):
        """Predicts a new color from saved models
        Parameters
        ----------
            color: numpy array of shape (1,3)
                A RGB color
        
        Returns
        -------
            str, predicted color class
        """
        if color.shape != (1,3):
            raise ValueError(f"{repr(color)} must be of shape (1,3)")
        return self.model.predict(color)[0]

    @staticmethod
    def preprocess(colors):
        new_colors = [np.array(color) for color in colors]
        for color in new_colors:
            color.shape = (1,3)
        return new_colors

    @staticmethod
    def simple_predict(color):
        """A simple predictor using Euclidean Distance
        Parameters
        ----------
            color: numpy array of shape (1,3)
                A RGB color
        
        Returns
        -------
            str, predicted color class
        """
        data = np.load(str(ColorGroup.LABELS_PATH), allow_pickle=True)
        samples = data.get("samples")
        labels = data.get("labels")

        euclid_dist = np.sqrt(np.sum((samples - color) ** 2, axis=1))
        idx = np.argmin(euclid_dist)
        return labels[idx]

    @staticmethod
    def kneighbors_predict(color, n_neighbors=10, weights="uniform"):
        """Builds a new KNearestNeighbor classifier
        Parameters
        ----------
            color: numpy array of shape (1,3)
                A RGB color
        
        Returns
        -------
            str, predicted color class
        """
        data = np.load(ColorGroup.LABELS_PATH)
        samples = data.get("samples")
        labels = data.get("labels")

        model = KNeighborsClassifier(n_neighbors, weights=weights)
        model.fit(samples, labels)
        return model.predict(color)[0]


# =====================================


def image_loader(fpath, source_dir, target_size=(150,150)):
    image = cv2.cvtColor(cv2.imread(f"{source_dir}/{fpath}.png"), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, cv2.INTER_CUBIC)
    return HighlightDetector.preprocess(image)

if __name__ == "__main__":
    # data is a dataframe in style of IAM dataset with row
    # row["id"] holds the image filename (with .png extension)
    # row["highlighted"] has "highlighted" value for highlighted words, "non-highlighted" otherwise
    # source_dir is the dir where all filenames are located

    X = np.zeros((len(data), *HighlightDetector.TARGET_SIZE))
    for idx, row in data.iterrows():
        X[idx, :, :, :] = image_loader(row["id"], SOURCE_DIR)

    Y = data["highlighted"]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, encoded_Y, test_size=0.2, shuffle=True)

    net = HighlightDetector()
    # net.cross_validate(X, encoded_Y)
    net.train(X_train, Y_train, plot=False)
    