from numpy.linalg import inv, det
from numpy import ndarray
from typing import Dict, List

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
        self.points = [] # [x,y]
        self.chunk_number = chunk_number

        self.above: Region = None
        self.below: Region = None
        self.valley_ids = np.array(())
        if initial_valley_id != -1: #means that there is a valley
            self.valley_ids = np.append(self.valley_ids, initial_valley_id)
            
        self.initial_valley_id = initial_valley_id
    
    def append_valley_id(self, new_valley_id):
        self.valley_ids = np.append(self.valley_ids, new_valley_id)

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
                    self.points.append((prev_row, j))

        # Add line points between the valleys
        
        for id in self.valley_ids:
            chunk_index = map_valley[id].chunk_index
            chunk_row = map_valley[id].position
            chunk_start_column = chunk_index * chunk_width

            for j in range(chunk_start_column, chunk_start_column + chunk_width):
                self.min_row_pos = min(self.min_row_pos, chunk_row)
                self.max_row_pos = max(self.max_row_pos, chunk_row)
                if c == j:
                    c += 1
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
                    c += 1
                    self.points.append((chunk_row, j))

    @staticmethod
    def get_min_row_position(line):
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
        self.histogram = np.array([0 for i in range(self.thresh_img.shape[0])])

        self.peaks: List[Peak] = [] # Peak type
        self.valleys: List[Valley] = [] #Valley type
        self.avg_height = 0
        self.avg_white_height = 0
        self.lines_count = 0
    
    def calculate_histogram(self, debug: bool):
        """"""
        self.thresh_img = cv2.medianBlur(self.thresh_img, 5) # get the smoothed profile via a median filter of size 5

        if debug:
            hist_init_func = Chunk.optimized_calc_histogram_init.py_func
            hist_calc_func = Chunk.optimized_calc_histogram.py_func
        else:
            hist_init_func = Chunk.optimized_calc_histogram_init
            hist_calc_func = Chunk.optimized_calc_histogram

        self.histogram, self.lines_count, self.avg_height, white_spaces, white_lines_count = hist_init_func(
            self.thresh_img, self.histogram, self.lines_count, self.avg_height
        )

        white_spaces = np.sort(white_spaces, kind="stable")

        self.avg_height, self.avg_white_height, self.lines_count  = hist_calc_func(
            white_spaces, self.avg_height, self.avg_white_height, white_lines_count, self.lines_count
        )
        

    @staticmethod
    @nb.njit(cache=True)
    def optimized_calc_histogram_init(thresh_image, histogram, lines_count, avg_height):
        rows, cols = thresh_image.shape[:2]

        current_height, current_white_count, white_lines_count = 0, 0, 0
        white_spaces = np.array([-1])

        for i in range(rows):
            black_count = 0

            for j in range(cols):
                if thresh_image[i, j] == 0:
                    black_count = black_count + 1
                    histogram[i] = histogram[i] + 1

            if black_count > 0:
                current_height = current_height + 1
                if current_white_count > 0:
                    white_spaces = np.append(white_spaces, current_white_count)
                    
                current_white_count = 0
            else:
                current_white_count = current_white_count + 1
                if current_height > 0:
                    lines_count = lines_count + 1
                    avg_height = avg_height + current_height

                current_height = 0
        
        return (histogram, lines_count, avg_height, white_spaces[1:], white_lines_count)

    @staticmethod
    @nb.njit(cache=True)
    def optimized_calc_histogram(white_spaces, avg_height, avg_white_height, white_lines_count, lines_count,
        MAX_AVG_HEIGHT = 30, AVG_HEIGHT_MULT1 = 4, AVG_HEIGHT_MULT2 = 1.5):
        

        # Compute the whitespaces' average height
        for i in range(len(white_spaces)):
            if white_spaces[i] > AVG_HEIGHT_MULT1 * avg_height:
                break

            avg_white_height = avg_white_height +  white_spaces[i]
            white_lines_count = white_lines_count + 1
        
        if white_lines_count:
            avg_white_height = avg_white_height / white_lines_count
        
        # Compute the average line height
        if lines_count > 0:
            avg_height = avg_height / lines_count
        
        avg_height = max(MAX_AVG_HEIGHT, int(avg_height * AVG_HEIGHT_MULT2))

        return (avg_height, avg_white_height, lines_count)

    def find_peaks_valleys(self, map_valley = {}, debug=False):
        self.calculate_histogram(debug)
        
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

    def update_region(self, gray_image, region_id, debug: bool):
        gray_height, gray_width = gray_image.shape[:2]

        self.region_id = region_id

        if self.top.initial_valley_id == -1:
            min_region_row = 0
            self.row_offset = 0
        else:
            min_region_row = self.top.min_row_pos
            self.row_offset = self.top.min_row_pos

        if self.bottom.initial_valley_id == -1:
            max_region_row = gray_height
        else:
            max_region_row = self.bottom.max_row_pos

        start = self.start = int(min(min_region_row, max_region_row))
        end = self.end = int(max(min_region_row, max_region_row))

        is_top_valley_ids_empty = len(self.top.valley_ids) == 0
        is_bottom_valley_ids_empty = len(self.bottom.valley_ids) == 0

        self.region = np.ones((end - start, gray_width), dtype=np.uint8) * 255

        # Pre-requisites for filling region
        region_points = np.array(
            [(self.get_new_point_start(col), self.get_new_point_end(col, gray_height - 1))
                for col in range(gray_width)]
        )

        if debug:
            fill_region_func = Region.fill_region.py_func
            calculate_mean_func = Region.calculate_mean.py_func
            calculate_covariance_func = Region.calculate_covariance.py_func
        else:
            fill_region_func = Region.fill_region
            calculate_mean_func = Region.calculate_mean
            calculate_covariance_func = Region.calculate_covariance

        self.region, self.height = fill_region_func(
            self.region,
            self.height,
            gray_image,
            min_region_row,
            region_points
        )

        self.mean = calculate_mean_func(self.region, self.row_offset, self.mean)
        self.covariance = calculate_covariance_func(self.region, self.row_offset, self.mean)

        return cv2.countNonZero(self.region) == (self.region.shape[0] * self.region.shape[1])
        
    def get_new_point_start(self, idx, value=0):
        if (self.top.valley_ids.shape[0] != 0) and len(self.top.points) != 0:
            value = self.top.points[idx][0]
        
        return value

    def get_new_point_end(self, idx, value):
        if (self.bottom.valley_ids.shape[0] != 0) and len(self.bottom.points) != 0:
            value = self.bottom.points[idx][0]
        
        return value

    @staticmethod
    @nb.jit(cache=True)
    def fill_region(region, region_height, gray_image, min_region_row, region_points):
        for col in range(gray_image.shape[1]):
            start, end = region_points[col]

            # Calculate region height
            if end > start:
                region_height = max(region_height, end - start)

            start, end = int(start), int(end)
            for i in range(start, end):
                region[i - int(min_region_row)][col] = gray_image[i][col]

        return (region, region_height)


    @staticmethod
    @nb.njit(cache=True)
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
