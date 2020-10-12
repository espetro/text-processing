from typing import List, Optional, Tuple, Union
from numpy import ndarray as NumpyArray

import matplotlib.pyplot as plt
import numpy as np

class Printer:
    def __init__(self, images: List[Union[np.ndarray, List]], titles: Optional[List] = None, figsize: Tuple = (10, 9)):
        if not isinstance(images, list) and not isinstance(images, NumpyArray):
            raise ValueError("Images must be in a list")

        self.y_size = len(images)
        self.x_size = 1            
        self.images = images
        self.titles = titles
        self.figsize = figsize
        
    def print_(self, nested_lists=False, only_vertical=False, flip=False):
        if nested_lists:
            self._nested_print(flip)
        else:
            self._simple_print(flip, only_vertical)
            
    def _nested_print(self, flip):
        y_size = self.y_size
        x_size = np.max([len(img_list) for img_list in self.images])
        if flip:
            tmp = y_size
            y_size = x_size
            x_size = tmp
            
        if not isinstance(self.images[0], list):
            images = np.array(self.images).flatten()
        else:
            images = []
            for image_list in self.images:
                images = images + image_list
        
        titles = self.titles or ["" for img in images]
        
        _, axes = plt.subplots(y_size, x_size, figsize=self.figsize)
        axes = axes.flatten()
        
        for i, (img, title) in enumerate(zip(images, titles)):
            cmap = "gray"
            if len(img.shape) == 3 and img.shape[-1] == 3:
                cmap = None

            axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(title)

        for j in range(0, y_size * x_size):
            axes[j].axis("off")
                
        plt.show()
    
    def _simple_print(self, flip, only_vertical, MAX_IMAGES=10):
        if len(self.images) == 1:
            cmap = "gray"
            if len(self.images[0].shape) == 3 and self.images[0].shape[-1] == 3:
                cmap = None

            plt.figure(figsize=self.figsize)
            plt.imshow(self.images[0], cmap=cmap)
            plt.axis("off")
        else:
            titles = self.titles or ["" for img in self.images]
            
            if len(self.images) > MAX_IMAGES and not only_vertical:
                self.y_size = self.y_size // 2
                self.x_size = len(self.images) // self.y_size
                if (self.x_size * self.y_size) < len(self.images):
                    self.x_size += len(self.images) - (self.x_size * self.y_size)

            if flip:
                y_size = self.x_size
                x_size = self.y_size
            else:
                y_size = self.y_size
                x_size = self.x_size
            
            _, axes = plt.subplots(y_size, x_size, figsize=self.figsize)
            
            if x_size != 1 and y_size != 1:
                axes = axes.flatten()
                
            for i, (img, title) in enumerate(zip(self.images, titles)):
                cmap = "gray"
                if len(img.shape) == 3 and img.shape[-1] == 3:
                    cmap = None

                axes[i].imshow(img, cmap=cmap)
                axes[i].set_title(title)
                axes[i].axis("off")

        plt.show()
