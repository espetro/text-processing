from numpy.linalg import inv, det
from typing import *

import numpy as np
import numba as nb
import math
import cv2
import sys

class Peak:
    """A class representing peaks (local maximum points in a histogram)"""
    def __init__(self, position=0, value=0):
        """
        position: row position
        value: the number of foreground pixels
        """
        self.position = position
        self.value = value

    def get_value(self, peak):
        """
        used to sort Peak lists based on value
        """
        return peak.value

    def get_row_position(self, peak):
        """
        used to sort Peak lists based on row position
        :param peak:  
        """
        return peak.position


class Valley:
    """A class representing valleys (local contiguous minimum points in a histogram)"""
    ID = 0

    def __init__(self, chunk_index=0, position=0):
        """
        chunk_index: The index of the chunk in the chunks vector
        position: The row position
        """
        self.valley_id = Valley.ID
        
        self.chunk_index = chunk_index
        self.position = position

        # Whether it's used by a line or not
        self.used = False
        # The line to which this valley is connected
        self.line = Line()

        Valley.ID += 1
    
    def compare_2_valley(self, v1, v2):
        #used to sort valley lists based on position
        return v1.position < v2.position


class Line:
    """Represent the separator among line regions."""
    
    def __init__(self, initial_valley_id=-1, chunk_number=20):
        """
        min_row_pos : the row at which a region starts
        max_row_pos : the row at which a region ends
        
        above: Region above the line
        below: Region below the line

        valley_ids: ids of valleys
        points: points representing the line
        """
        self.min_row_pos = 0
        self.max_row_pos = 0
        self.points = [] # (x,y)
        self.chunk_number = chunk_number

        self.above = 0 #Region type
        self.below = 0 #Region type
        self.valley_ids = []
        if initial_valley_id != -1: #means that there is a valley
            self.valley_ids.append(initial_valley_id)
            
        self.initial_valley_id = initial_valley_id
    
    def generate_initial_points(self, chunk_width, img_width, map_valley={}):
        """"""
        c, prev_row = 0, 0
        
        #sort the valleys according to their chunk number
        self.valley_ids.sort()

        #add line points in the first chunks having no valleys
        if map_valley[self.valley_ids[0]].chunk_index > 0:
            prev_row = map_valley[self.valley_ids[0]].position
            self.max_row_pos = self.min_row_pos = prev_row
            for j in range(map_valley[self.valley_ids[0]].chunk_index * chunk_width):
                if c == j:
                    c += 1
                    self.points.append((prev_row,j))

        # Add line points between the valleys
        
        for id in self.valley_ids:
            chunk_index = map_valley[id].chunk_index
            chunk_row = map_valley[id].position
            chunk_start_column = chunk_index * chunk_width

            for j in range(chunk_start_column, chunk_start_column + chunk_width):
                self.min_row_pos = min(self.min_row_pos, chunk_row)
                self.max_row_pos = max(self.max_row_pos, chunk_row)
                if c == j:
                    c+=1
                    self.points.append((chunk_row, j))
        
            if prev_row != chunk_row:
                prev_row = chunk_row
                self.min_row_pos = min(self.min_row_pos, chunk_row)
                self.max_row_pos = max(self.max_row_pos, chunk_row)

        # Add line points in the last chunks having no valleys
        if self.chunk_number - 1 > map_valley[self.valley_ids[-1]].chunk_index:
            chunk_index = map_valley[self.valley_ids[-1]].chunk_index
            chunk_row = map_valley[self.valley_ids[-1]].position

            for j in range(chunk_index * chunk_width + chunk_width,img_width):
                if c == j:
                    c+=1
                    self.points.append((chunk_row, j))

        
    def get_min_row_position(self, line):
        return line.min_row_pos


class Chunk:
    """Class Chunk represents the vertical segment cut.
    There are 20 CHUNK, because each every chunk is 5% of a image
    """

    def __init__(self, index = 0, start_col = 0, width = 0, img = np.array(())):
        """
        index: index of the chunk
        start_col: the start column positition
        width: the width of the chunk
        img: gray iamge
        histogram: the value of the y histogram projection profile
        peaks: found peaks in this chunk
        valleys: found valleys in this chunk
        avg_height: average line height in this chunk
        avg_white_height: average space height in this chunk
        lines_count: the estimated number of lines in this chunk
        """
        self.index = index
        self.start_col = start_col
        self.width = width
        self.thresh_img = img.copy()

        # length is the number of rows in an image
        self.histogram = [0 for i in range(self.thresh_img.shape[0])]

        self.peaks = [] # Peak type
        self.valleys = [] #Valley type
        self.avg_height = 0
        self.avg_white_height = 0
        self.lines_count = 0
    
    def calculate_histogram(self):
        # get the smoothed profile by applying a median filter of size 5
        cv2.medianBlur(self.thresh_img, 5, self.thresh_img)

        current_height = 0
        current_white_count = 0
        white_lines_count = 0

        white_spaces = []
        
        rows, cols = self.thresh_img.shape[:2]

        for i in range(rows):
            black_count = 0
            for j in range(cols):
                if self.thresh_img[i][j] == 0:
                    black_count += 1
                    self.histogram[i] += 1
            if black_count:
                current_height += 1
                if current_white_count:
                    white_spaces.append(current_white_count)
                current_white_count = 0
            else:
                current_white_count += 1
                if current_height:
                    self.lines_count += 1
                    self.avg_height += current_height
                current_height = 0

        #calculate the white spaces average height
        white_spaces.sort()  # sort ascending
        for i in range(len(white_spaces)):
            if white_spaces[i] > 4 * self.avg_height:
                break
            self.avg_white_height += white_spaces[i]
            white_lines_count+=1
        
        if white_lines_count:
            self.avg_white_height /= white_lines_count
        #calculate the average line height
        if self.lines_count:
            self.avg_height /= self.lines_count
        
        # 30 is hyper-parameter
        self.avg_height = max(30, int(self.avg_height + self.avg_height / 2.0))

    # @nb.jit(forceobj=True, cache=True)
    def find_peaks_valleys(self, map_valley = {}):
        self.calculate_histogram()
        #detect peaks
        len_histogram = len(self.histogram)

        for i in range(1, len_histogram - 1):
            left_val = self.histogram[i - 1]
            centre_val = self.histogram[i]
            right_val = self.histogram[i + 1]
            #peak detection
            if centre_val >= left_val and centre_val >= right_val:
                # Try to get the largest peak in same region.
                if len(self.peaks) != 0 and i - self.peaks[-1].position <= self.avg_height // 2 and centre_val >= self.peaks[-1].value:
                    self.peaks[-1].position = i
                    self.peaks[-1].value = centre_val
                elif len(self.peaks) > 0 and i - self.peaks[-1].position <= self.avg_height // 2 and centre_val < self.peaks[-1].value:
                    abc = 0
                else:
                    self.peaks.append(Peak(position=i, value=centre_val))
        
        peaks_average_values = 0
        new_peaks = []  # Peak type
        for p in self.peaks:
            peaks_average_values += p.value
        peaks_average_values //= max(1, int(len(self.peaks)))

        for p in self.peaks:
            if p.value >= peaks_average_values / 4:
                new_peaks.append(p)
        
        self.lines_count = int(len(new_peaks))

        self.peaks = new_peaks
        #sort peaks by max value and remove the outliers (the ones with less foreground pixels)
        self.peaks.sort(key=Peak().get_value)
        #resize self.peaks
        if self.lines_count + 1 <= len(self.peaks):
            self.peaks = self.peaks[:self.lines_count + 1]
        else:
            self.peaks = self.peaks[:len(self.peaks)]
        self.peaks.sort(key=Peak().get_row_position)

        #search for valleys between 2 peaks
        for i in range(1, len(self.peaks)):
            min_pos = (self.peaks[i - 1].position + self.peaks[i].position) / 2
            min_value = self.histogram[int(min_pos)]
            
            start = self.peaks[i - 1].position + self.avg_height / 2
            end = 0
            if i == len(self.peaks):
                end = self.thresh_img.shape[0]  #rows
            else:
                end = self.peaks[i].position - self.avg_height - 30

            for j in range(int(start), int(end)):
                valley_black_count = 0
                for l in range(self.thresh_img.shape[1]):  #cols
                    if self.thresh_img[j][l] == 0:
                        valley_black_count += 1
                
                if i == len(self.peaks) and valley_black_count <= min_value:
                    min_value = valley_black_count
                    min_pos = j
                    if min_value == 0:
                        min_pos = min(self.thresh_img.shape[0] - 10, min_pos + self.avg_height)
                        j = self.thresh_img.shape[0]
                elif min_value != 0 and valley_black_count <= min_value:
                    min_value = valley_black_count
                    min_pos = j
            
            new_valley = Valley(chunk_index=self.index, position=min_pos)
            self.valleys.append(new_valley)
            
            # map valley
            map_valley[new_valley.valley_id] = new_valley
        return int(math.ceil(self.avg_height))


# @nb.jitclass()
class Region():
    """Class representing the line regions"""

    def __init__(self, top=Line(), bottom=Line()):
        """
        region_id: region's id
        region: 2d matrix representing the region
        top: Lines representing region top boundaries
        bottom: Lines representing region bottom boundaries
        height: Region's height
        row_offset: the offset of each col to the original image matrix
        covariance:
        mean: The mean of
        """
        self.top = top
        self.bottom = bottom

        self.region_id = 0
        self.height = 0

        self.region = np.array(()) # used for binary image
        self.start, self.end = None, None
        
        self.row_offset = 0
        self.covariance = np.zeros([2, 2], dtype=np.float32)
        self.mean = np.zeros((1, 2))

    def update_region(self, gray_image, region_id):
        self.region_id = region_id

        if self.top.initial_valley_id == -1:  # none
            min_region_row = 0
            self.row_offset = 0
        else:
            min_region_row = self.top.min_row_pos
            self.row_offset = self.top.min_row_pos

        if self.bottom.initial_valley_id == -1:  # none
            max_region_row = gray_image.shape[0]  # rows
        else:
            max_region_row = self.bottom.max_row_pos

        start = self.start = int(min(min_region_row, max_region_row))
        end = self.end = int(max(min_region_row, max_region_row))

        # print((start, end))

        self.region = np.ones((end - start, gray_image.shape[1]), dtype=np.uint8) * 255

        # Fill region.
        for c in range(gray_image.shape[1]):
            if len(self.top.valley_ids) == 0:
                start = 0
            else:
                if len(self.top.points) != 0:
                    start = self.top.points[c][0]
            if len(self.bottom.valley_ids) == 0:
                end = gray_image.shape[0] - 1
            else:
                if len(self.bottom.points) != 0:
                    end = self.bottom.points[c][0]

            # Calculate region height
            if end > start:
                self.height = max(self.height, end - start)

            for i in range(int(start), int(end)):
                self.region[i - int(min_region_row)][c] = gray_image[i][c]

        self.mean = Region.calculate_mean(self.region, self.row_offset, self.mean)
        self.covariance = Region.calculate_covariance(self.region, self.row_offset, self.mean)

        return cv2.countNonZero(self.region) == (self.region.shape[0] * self.region.shape[1])

    @staticmethod
    @nb.jit()
    def calculate_mean(region, row_offset, mean):
        mean[0][0] = 0.0
        mean[0][1] = 0.0
        n = 0

        reg_height, reg_width = region.shape[:2]
        for i in range(reg_height):
            for j in range(reg_width):
                # if white pixel continue.
                if region[i][j] == 255.0:
                    continue
                if n == 0:
                    n = n + 1
                    mean[0][0] = i + row_offset
                    mean[0][1] = j
                else:
                    vec = np.zeros((1,2))
                    vec[0][0] = i + row_offset
                    vec[0][1] = j
                    mean = ((n - 1.0) / n) * mean + (1.0 / n) * vec
                    n = n + 1
        
        return mean

    @staticmethod
    @nb.njit(cache=True)
    def calculate_covariance(region, row_offset, mean):
        # Total number of considered points (pixels) so far
        n = 0
        reg_height, reg_width = region.shape[:2]

        covariance = np.zeros((2, 2))
        sum_i_squared = 0
        sum_j_squared = 0
        sum_i_j = 0

        for i in range(reg_height):
            for j in range(reg_width):
                # if white pixel continue
                if int(region[i][j]) == 255:
                    continue

                new_i = i + row_offset - mean[0][0]
                new_j = j - mean[0][1]

                sum_i_squared += new_i * new_i
                sum_i_j += new_i * new_j
                sum_j_squared += new_j * new_j
                n += 1

        if n:
            covariance[0][0] = float(sum_i_squared / n)
            covariance[0][1] = float(sum_i_j / n)
            covariance[1][0] = float(sum_i_j / n)
            covariance[1][1] = float(sum_j_squared / n)

        return covariance

    @staticmethod
    @nb.njit(cache=True)
    def bi_variate_gaussian_density(point, mean, covariance):
        point[0][0] -= mean[0][0]
        point[0][1] -= mean[0][1]

        point_transpose = np.transpose(point)
        ret = ((point * inv(covariance) * point_transpose))
        ret *= np.sqrt(det(2 * math.pi * covariance))

        return int(ret[0][0])
