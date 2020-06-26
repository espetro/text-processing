from preproc import RobustBinarize
from numpy import ndarray

import pandas as pd
import numpy as np
import h5py as h5
import cv2
import os

class DataUnpack:
    """A set of functions to load Image datasets in HDF5 format into ImageDataGenerator Keras objects.
    It works along with the TinyData class.
    
    Source:
        https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/preproc.py
        https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/generator.py
        https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/evaluation.py
        https://machinelearnings.co/deep-spelling-9ffef96a24f6#.2c9pu8nlm
    """
    @staticmethod
    def imwrite(fpath, image, color=False):
        if color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fpath, image)
        else:
            cv2.imwrite(fpath, image)

    @staticmethod
    def resize(name, image: ndarray, target_size, aspect_ratio: float=None):
        """Resizes a grayscale image

        image: numpy ndarray with shape (height, width) and 0 channels

        Source:
            https://stackoverflow.com/questions/58248121/opencv-python-how-to-overlay-an-image-into-the-centre-of-another-image
            https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
        """
        h, w = image.shape
        image_ratio = np.round(w / h, 2)
        new_image = image.copy()

        if aspect_ratio:
            ratio_diff = abs(np.round(aspect_ratio, 2) - image_ratio)
            if ratio_diff > 0.1:
                mult_factor = aspect_ratio / image_ratio
                resized_w = int(mult_factor * w)
                resized_h = h
                
                if resized_w < w: # as we're padding the image, we don't want to crop it
                    # so instead of resizing width, it resizes height
                    mult_factor = image_ratio / aspect_ratio
                    resized_w = w
                    resized_h = int(mult_factor * h)

                new_image = np.ones((resized_h, resized_w), np.uint8) * 255
                hback, wback = new_image.shape
                
                yoff, xoff = ((hback - h) // 2, (wback - w) // 2)
                new_image[yoff:(yoff + h), xoff:(xoff + w)] = image

        mean = np.sum(new_image.shape[:2]) // 2
        maxi = np.max(new_image.shape[:2])
        interpolation = cv2.INTER_AREA if maxi > mean else cv2.INTER_CUBIC
        return cv2.resize(new_image, target_size, interpolation)

    @staticmethod
    def unpack_set(set_name, dest_dir, file, color=False, target_size=None, aspect_ratio=None, max_word_length=34):
        """Performs the unpacking step for each dataset split (train/test/validation)"""
        vector_labels = [f"X{i}" for i in range(max_word_length)]
        columns = ["name", "label", "fname"] + vector_labels
        result = pd.DataFrame(columns=columns)

        os.makedirs(f"{dest_dir}/{set_name}", exist_ok=True)

        for sample_name in file[set_name]:
            data = file[f"{set_name}/{sample_name}"]
            image = data[:]
            label = data.attrs["label"].decode("utf-8")
            encoded_vector = data.attrs["intlabel"][:]  # int-encoded label
            
            if target_size and aspect_ratio: # are not None
                image = DataUnpack.resize(sample_name, image, target_size, aspect_ratio)

            # Remove cursive style (WORKS BADLY FOR 'train_spa/w02_bw_22' and takes so much time)
            # image = RobustBinarize.remove_cursive_style(image)

            # transpose the image (as numpy uses (h,w) shapes and tf uses (w, h) shapes)
            image = image.transpose()

            # at the time of writing, it should follow the folder structure for RIMES and IAM
            DataUnpack.imwrite(f"{dest_dir}/{set_name}/{sample_name}.png", image, color=color)

            # add a dataframe row for storing image data
            idx = len(result)
            result.loc[idx] = [sample_name, label, f"{label}.png", *encoded_vector]
            # result.iloc[idx, 0] = sample_name
            # result.iloc[idx, 1] = label
            # result.iloc[idx, 2] = f"{label}.png"
            # result.iloc[idx, 3:] = encoded_vector

        return result

    @staticmethod
    def load_set(set_name, file, target_size=None, aspect_ratio=None):
        """Performs the unpacking step for each dataset split (train/test/validation). Images are restored as RGB."""
        images, labels, vectors = [], [], []

        for sample_name in file[set_name]:
            data = file[f"{set_name}/{sample_name}"]
            image = data[:]
            label = data.attrs["label"].decode("utf-8")
            encoded_vector = data.attrs["intlabel"][:]  # int-encoded label
            
            if target_size and aspect_ratio: # are not None
                image = DataUnpack.resize(sample_name, image, target_size, aspect_ratio)

            # transpose the image (as numpy uses (h,w) shapes and tf uses (w, h) shapes)
            image = image.transpose()
            image = np.expand_dims(image, axis=-1)  # add a sigle, grayscale channel to the image

            images.append(image)
            labels.append(label)
            vectors.append(encoded_vector)

        return [np.asarray(images), np.asarray(labels), np.asarray(vectors)]  # so that they can be modified

    @staticmethod
    def unpack(input_path, dest_dir, color=False, save_to_disk=False, target_size=None, aspect_ratio:float=None, max_word_length=34):
        """Deminifies / Unpacks the compressed file into train/test/validation objects.
        
        Parameters
        ----------
            ...

            save_to_disk: bool, default False.
                If True, saves the images and labels to disk (in .png and .csv formats). Then returns the .csv objects.
                If False, loads the images and labels to numpy arrays, and returns them as objects.

            ...
        """
        dataset_name = input_path.replace(".h5", "").split("/")[-1]

        with h5.File(input_path, "r") as f:
            if save_to_disk:    
                train = DataUnpack.unpack_set("train", dest_dir, f, color, target_size, aspect_ratio, max_word_length)
                train.to_csv(f"{dest_dir}/train/train_{dataset_name}.csv", sep=",", index=False)
                
                test = DataUnpack.unpack_set("test", dest_dir, f, color, target_size, aspect_ratio, max_word_length)
                test.to_csv(f"{dest_dir}/test/test.csv_{dataset_name}", sep=",", index=False)

                # if there is a valid split stored, then extract it as well
                if f.get("valid", None):
                    valid = DataUnpack.unpack_set("valid", dest_dir, f, color, target_size, aspect_ratio, max_word_length)
                    valid.to_csv(f"{dest_dir}/valid/valid_{dataset_name}.csv", sep=",", index=False)
                else:
                    valid = None
            else:
                train = DataUnpack.load_set("train", f, target_size, aspect_ratio)
                test = DataUnpack.load_set("test", f, target_size, aspect_ratio)
                
                # if there is a valid split stored, then extract it as well
                valid = None
                if f.get("valid", None):
                    valid = DataUnpack.load_set("valid", f, target_size, aspect_ratio)
        
        return train, test, valid
