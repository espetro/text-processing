from sklearn.neighbors import KNeighborsClassifier

import pkg_resources
import pickle as pk
import numpy as np

class ColorGroup:
    """
    Represents the color naming algorithm.
    Given an RGB tuple, it returns the name of the closest CSS3 name color.
    It uses K-NN under the hood.

    Source:
        https://blog.algolia.com/how-we-handled-color-identification/
        https://blog.xkcd.com/2010/05/03/color-survey-results/
    """
    # KNearestNeighbor with K=10 and uniform weights
    MODEL_PATH = pkg_resources.resource_filename("tfgpkg.recognition.data", "knn_10_uniform_white_mini.pk")
    LABELS_PATH = pkg_resources.resource_filename("tfgpkg.recognition.data", "color_names_white_mini.npz")
    
    COLORS = ['black', 'green', 'blue', 'brown', 'purple', 'maroon', 'red', 'pink', 'orange', 'yellow', 'gold', 'white']

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
            raise ValueError(f"{repr(color)} must be of shape (1,3) ", color)
        return self.model.predict(color)[0]

    @staticmethod
    def preprocess(colors):
        new_colors = [np.array(color) for color in colors]
        for color in new_colors:
            color.shape = (1,3)
        return new_colors

    @staticmethod
    def to_css(color_name):
        """Validates if the color name belongs to the CSS1/2/3 spec"""
        return {"mustard": "gold"}.get(color_name, color_name)

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

    @staticmethod
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
