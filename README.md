This folder contains all packages developed and used for the thesis.


```python
from preproc import RobustBinarize, ColorGroup, Quantize
from time import time

import cv2

image = cv2.cvtColor(cv2.imread("./samples/spanish-words-mini/w07_rgb_17.png"), cv2.COLOR_BGR2RGB)

# this line will take more (as numba compiles down the code)
curr = time()
binarized = RobustBinarize.illumination_compensation(image)
print("Elapsed time:", time() - curr)

# this line will take less than a second (now that the algorithm is compiled down to native!)
curr = time()
binarized2 = RobustBinarize.illumination_compensation(image)
print("Elapsed time:", time() - curr)

classifier = ColorGroup()
color = np.array([[255, 0, 0]])  # array of shape (1,3)
label = classifier.predict(color)
print(f"{color} label is {label}")

reduced_image = Quantize.reduce_palette(image, num_colors=4)
```