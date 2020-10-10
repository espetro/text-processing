from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

class Printer:
    def __init__(self, images: List[Union[np.ndarray, List]], titles: Optional[List] = None, figsize: Tuple = (10, 9)):
        self.y_size = len(images)
        self.x_size = 1            
        self.images = images
        self.titles = titles
        self.figsize = figsize
        
    def print_(self, nested_lists=False, flip=False):
        if nested_lists:
            self._nested_print(flip)
        else:
            self._simple_print(flip)
            
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
    
    def _simple_print(self, flip):        
        titles = self.titles or ["" for img in self.images]
    
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
