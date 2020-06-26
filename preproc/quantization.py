from sklearn.cluster import KMeans
from sklearn.utils import shuffle

import numpy as np

class Quantize:
    """
    Image quantization functions.
    """
    @staticmethod
    def reduce_palette(image, num_colors=3):
        """Reduces the color palette from a given image.
        
        Parameters
        ----------
            image: numpy array
                A 3D array containing an RGB/Grayscale image
            num_colors: int, default 3
                Number of colors to obtain after quantization
        
        Returns
        -------
            image (numpy array) with quantized colors
        """
        w, h, _ = image.shape
        image = Quantize.flatten(image)
        # Fit the model on a data sub-sample
        sample = shuffle(image, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(sample)
        # Predict color indices on the full image
        labels = kmeans.predict(image)
        
        return Quantize.recreate(
            kmeans.cluster_centers_,
            labels,
            w,
            h
        )

    @staticmethod
    def flatten(image):
        """Turn 3D into 2D array."""
        w, h, ch = image.shape
        return image.reshape(w * h, ch)

    @staticmethod
    def normalize(img):
        """Rescale image values from [0..255] range to [0..1]."""
        return img / 255.

    @staticmethod
    def recreate(centers, labels, w, h):
        """Recreate the compressed image from the cluster centers and quantized colors."""
        ch = centers.shape[1]
        image = np.zeros((w, h, ch))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = centers[labels[label_idx]]
                label_idx += 1
        return Quantize.normalize(image)

