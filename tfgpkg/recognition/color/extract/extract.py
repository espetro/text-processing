from sklearn.preprocessing import LabelEncoder
from recognition.colorthief import ColorThief
from PIL import Image

import numpy as np
import numbers
import cv2


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
    