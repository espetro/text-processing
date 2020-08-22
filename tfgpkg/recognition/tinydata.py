from recognition import StringVectorizer
from pandas import DataFrame as DF
from tqdm import tqdm

import pandas as pd
import numpy as np
import h5py as h5
import cv2
import os

class TinyData:
    """A set of functions to minify HTR word image datasets into HDF5 files.
    It saves the text labels as one-hot vectors.
    """

    @staticmethod
    def imread(fpath:str, color=False):
        if color:
            return cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
        else:
            return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        
    @staticmethod
    def minify_set(compressed_file, subset_name, subset, x_label, y_label, name_label, source_dir, color=False, max_word_length=34):
        vectorizer = StringVectorizer(max_word_length)

        for _, row in tqdm(subset.iterrows()):
                fpath = f"{source_dir}/{row[x_label]}"
                image = TinyData.imread(fpath, color=color)

                label = row[y_label].encode("utf-8")  # turns Unicode into ASCII bytes
                nm = row[name_label]
                
                if image is None:
                    raise ValueError(f"({subset_name} Set) Image {fpath} has size 0. Please remove it from the dataset.")
                    
                new_data = compressed_file.create_dataset(
                    f"{subset_name}/{nm}", data=image, dtype=image.dtype, compression="gzip", compression_opts=9
                )
                new_data.attrs["label"] = label
                new_data.attrs["intlabel"] = vectorizer.vectorize(row[y_label])  # array of length 'max_word_length'
                

    @staticmethod
    def minify(train, test, valid, source_dir, target_path, name_label, x_label="fpath", y_label="word", color=False, max_word_length=34):
        """Minifies the training/test/validation splits in the target_path file.
        It needs the filepath column label and the Y class column label.
        """

        with h5.File(target_path, "w") as hf:
            
            TinyData.minify_set(hf, "train", train, x_label, y_label, name_label, source_dir, color, max_word_length)
            TinyData.minify_set(hf, "test", test, x_label, y_label, name_label, source_dir, color, max_word_length)

            if valid is not None:
                TinyData.minify_set(hf, "valid", valid, x_label, y_label, name_label, source_dir, color, max_word_length)


        print("Done")


