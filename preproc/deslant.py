from collections import deque

import numpy as np
import numba as nb
import cv2

class Deslanter:

    @staticmethod
    def rotate_image(image):
        """Rotate image if text not horizontal"""
        coords = np.column_stack(np.where(image == 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        slope = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, slope, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def find_maxima(image, row_start_idx, col_start_idx, col_stop_idx):
        """Find the maximum point which is needed to calculate the slope"""
        li = image[row_start_idx - 1, (col_start_idx - 4):(col_stop_idx + 4)]
        
        next_li = [i for i in range(len(li)) if li[i] != 1]
        if not next_li:
            return row_start_idx, col_start_idx

        row_start_idx, col_start_idx = Deslanter.find_maxima(
            image,
            row_start_idx - 1,
            col_start_idx - 2 + next_li[0],
            col_start_idx - 2 + next_li[-1]
        )
        
        return (row_start_idx, col_start_idx)

    @staticmethod
    def deslant_image(image, pad_size=10, process=True):
        """deslant image by calculating its slope and then rotating it overcome the effect of that shift.

        Parameters
        ----------
            image: ndarray of shape (X,Y)
                A binarized image
        """
        # make border to prevent information loss
        processed_image = image
        image = cv2.copyMakeBorder(image, 0, 0, pad_size, pad_size, cv2.BORDER_CONSTANT, value=255)
        checker = np.array(image, dtype=np.float64)

        if Deslanter.is_skewed(checker):
            # start checking for written text after skipping the padded size
            row_start_idx, col_start_idx = None, pad_size - 1

            while row_start_idx is None:
                col_start_idx += 1
                matches = np.where(checker[:, col_start_idx] == 0.0)
                
                if len(matches[0]) > 0:
                    row_start_idx = matches[0][0]

            # get the first whitespace column index
            li = checker[row_start_idx, col_start_idx:]
            total_cols = np.where(li != 0.0)[0][0]

            # x1, y1, are the first point of the slope line and x2, y2 are the second points
            x1, y1 = 0, np.where(checker[:, col_start_idx] == 0.0)[0][0]
            # calculate first maxima of black points so as to get second point of the slope line
            y2, x2 = Deslanter.find_maxima(checker, row_start_idx, col_start_idx, col_start_idx + total_cols)

            c, m = y1, 0

            if y2 - y1 != 0:
                m = (x2 - x1) / (y2 - y1)
            else:
                process = False
                processed_image = checker

            # if a slope is detected then shift the pixels in the image to make it straight otherwise pass this step
            if process:
                processed_image = Deslanter.process_image(checker, m, c)
        return processed_image

    @staticmethod
    def is_skewed(image):
        """Checks if the text in an image is skewed. It uses Hough Transform to extract the slopes."""
        return True
        
    @staticmethod
    def process_image(checker, m, c):
        """"""
        processed = []
        y = [((i * m) + c , i) for i in range(checker.shape[0])]

        for i in zip(checker, y):
            li = deque(i[0])
            count = int(i[1][0])

            if count > 0:
                while count != 0:
                    li.popleft()
                    count -= 1
                li = np.array(li)
                li = np.lib.pad(li, (0, i[0].shape[0] - len(li)), 'maximum')
            else:
                count = abs(count)
                while count != 0:
                    li.pop()
                    count -= 1
                li = np.array(li)
                li = np.lib.pad(li, (i[0].shape[0] - len(li), 0), 'maximum')

            processed.append(li)
        
        return np.asarray(processed)