from numba.core.errors import NumbaPendingDeprecationWarning
from numpy import ndarray as NumpyArray
from numbers import Integral
from tqdm import tqdm
from enum import Enum
from typing import List, Tuple

from tfgpkg.preproc.line.projection import *

import numpy as np
import numba as nb
import warnings
import logging
import cv2
import sys

logging.basicConfig(level=logging.INFO)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

# every chunk has a 5% of the width
CHUNK_NUMBER = 20
CHUNK_TOBE_PROCESSED = 5

@nb.njit(cache=True)
def union(a,b):
    """
    :param a: rect
    :param b: rect
    :return: union area 
    """
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

@nb.njit(cache=True)
def intersection(a,b):
    """
    :param a:  rect
    :param b:  rect
    :return: intersection area 
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y

    if w < 0 or h < 0:
        return None
    else:
        return [x, y, w, h]

@nb.njit(cache=True)
def mergeRects(rects):
    """
     rects = [(x,y,w,h),(x,y,w,h),...]
     return merged rects
    """
    merged_rects = []

    for i in range(len(rects)):
        is_repeated = False
        
        for j in range(i + 1, len(rects)):
            rect_tmp = intersection(rects[i], rects[j])

            # Check for intersection / union
            if rect_tmp is None:
                continue
            
            rect_tmp_area = rect_tmp[2] * rect_tmp[3]
            rect_i_area = rects[i][2] * rects[i][3]
            rect_j_area = rects[j][2] * rects[j][3]

            if ((rect_tmp_area == rect_i_area) or (rect_tmp_area == rect_j_area)):
                is_repeated = True
                merged_rect = union(rects[i], rects[j]) # Merging
                
                # Push in merged rectangle after checking all the inner loop
                if j == len(rects) - 2:
                    merged_rects.append(merged_rect)

                # Update the current vector
                rects[j] = merged_rect

        if is_repeated == False:
            merged_rects.append(rects[i])

    return merged_rects

class LineSegmentation:
    def __init__(self, image, scale_factor=2.0):
        """
        img: binarized image loaded
        output_path:
        chunks: The image chunks list
        lines_region: All the regions the lines found
        avg_line_height: the average height of lines in the image
        """
        self.logger = logging.getLogger("lineSegm")
        self.output_path = ""
        self.chunks = []
        self.chunk_width = 0
        self.map_valley = {}  # {int, Valley}
        self.predicted_line_height = 0

        self.initial_lines: List[Line] = []  # Line type
        self.lines_region: List[Region] = []  # region type

        self.avg_line_height = 0

        self.primes = LineSegmentation.sieve()

        self.img = image

        if len(image.shape) == 3:  # i.e. RGB image or Grayscale with a single channel
            if image.shape[-1] == 1:
                self.gray_img = self.img[:,:,0]
            else:
                self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = self.img

    @staticmethod
    def rescale(image, scale_factor):
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    def find_lines(self, debug=False, strip_whites=True, buffer_lines: int = 10, whitest_range: int = 10):
        """Return a list of cut images's path plus their bounding boxes"""
        self.pre_process_img()
        self.find_contours(debug)
        # Divide image into vertical chunks
        self.generate_chunks()
        # Get initial lines
        self.get_initial_lines(debug)

        # Get initial line regions
        self.generate_regions(debug)
        # repair all initial lines and generate the final line region
        self.repair_lines()
        # Generate the final line regions
        self.generate_regions(debug)

        _, width = self.img.shape[:2]
        return self.get_regions(strip_whites, width, buffer_lines, whitest_range)

    @staticmethod
    def sieve():
        """Performs the Sieve of Erastothenes."""
        not_primes_list = np.zeros(100007)
        primes = np.array([0])
        primes = primes[:0]

        not_primes_list[0] = not_primes_list[1] = 1
        for i in range(2, 100000):
            if not_primes_list[i] == 1:
                continue

            primes = np.append(primes, i)
            for j in range(i * 2, 100000, i):
                not_primes_list[j] = 1

        return primes

    @staticmethod
    def add_primes_to_list(primes, n, prob_primes):
        for i in range(len(primes)):
            while (n % primes[i]):
                n /= primes[i]
                prob_primes[i] += 1
                

    def pre_process_img(self):
        # blur image
        smoothed_img = cv2.blur(self.gray_img, (3, 3), anchor=(-1, -1))

        _, self.thresh = cv2.threshold(smoothed_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    def find_contours(self, debug: bool):
        contours = 0
        contours, hierachy = cv2.findContours(
            image=self.thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE, offset=(0,0)
        )

        bounding_rect = []
        for c in contours:
            cv2.approxPolyDP(c, 1, True, c)  # apply approximation to polygons
            bounding_rect.append(cv2.boundingRect(c))

        del(bounding_rect[-1])  # coz the last one is a contour containing whole image

        # check intersection among rects
        if debug:
            merged_rect = mergeRects.py_func(bounding_rect)
        else:
            merged_rect = mergeRects(bounding_rect)

        self.contours = merged_rect  # save contours AS rectangles

    def generate_chunks(self):
        rows, cols = self.thresh.shape[:2]
        width = cols
        self.chunk_width = int(width / CHUNK_NUMBER)

        start_pixel = 0
        for i_chunk in range(CHUNK_NUMBER):
            c = Chunk(index=i_chunk, start_col=start_pixel, width=self.chunk_width,
                      img = self.thresh[0:rows, start_pixel:start_pixel + self.chunk_width].copy())
            self.chunks.append(c)

            # cv2.imwrite(self.output_path + "/Chunk" + str(i_chunk) + ".jpg", self.chunks[-1].gray_img)

            start_pixel += self.chunk_width

    def connect_valleys(self, i, current_valley, line, valleys_min_abs_dist):
        """
        i: the chunk's index
        current_valley
        line
        valleys_min_abs_dist:
        """
        if (i <= 0 or len(self.chunks[i].valleys)==0):
            return line
        # choose the closest valley in right chunk to the start valley
        connected_to = -1
        min_dist = sys.maxsize
        # found valleys's size in this chunk
        valleys_size = len(self.chunks[i].valleys)
        for j in range(valleys_size):
            valley = self.chunks[i].valleys[j]
            # Check if the valley is not connected to any other valley
            if valley.used is True:
                continue

            dist = abs(current_valley.position - valley.position)

            if min_dist > dist and dist <= valleys_min_abs_dist:
                min_dist = dist
                connected_to = j

        # Return line if the current valley is not connected any more to a new valley in the current chunk of index i.
        if connected_to == -1:
            return line

        line.append_valley_id(self.chunks[i].valleys[connected_to].valley_id)

        v = self.chunks[i].valleys[connected_to]
        v.used = True
        return self.connect_valleys(i - 1, v, line, valleys_min_abs_dist)

    def get_initial_lines(self, debug: bool):
        """Generates a new set of initial lines"""
        number_of_heights, valleys_min_abs_dist = 0, 0

        # Get the histogram of the first CHUNK_TO_BE_PROCESSED and get the overall average line height.
        for i in range(CHUNK_TOBE_PROCESSED):
            self.avg_height = self.chunks[i].find_peaks_valleys(self.map_valley, debug)

            if self.avg_height:
                number_of_heights += 1
            valleys_min_abs_dist += self.avg_height

        valleys_min_abs_dist /= number_of_heights
        self.predicted_line_height = valleys_min_abs_dist

        # Start from the CHUNK_TOBE_PROCESSED chunk
        for i in range(CHUNK_TOBE_PROCESSED - 1, 0, -1):
            if len(self.chunks[i].valleys) == 0:
                continue

            # Connect each valley with the nearest ones in the left chunks.
            for valley in self.chunks[i].valleys:
                if valley.used is True:
                    continue

                # Start a new line having the current valley and connect it with others in the left.
                valley.used = True

                new_line = Line(valley.valley_id)
                new_line = self.connect_valleys(i - 1, valley, new_line, valleys_min_abs_dist)
                new_line.generate_initial_points(self.chunk_width, self.img.shape[1], self.map_valley)

                if len(new_line.valley_ids) > 1:
                    self.initial_lines.append(new_line)

    def generate_regions(self, debug: bool, MULT_1=2.5):
        """Generate a new set of regions given the current set of initial_lines"""
        if len(self.initial_lines) == 0:
            return

        self.initial_lines.sort(key=Line.get_min_row_position)
        
        # Get a new set of regions
        first_region = Region(bottom=self.initial_lines[0])
        first_region.update_region(self.gray_img, 0, debug)
        
        self.initial_lines[0].above = first_region

        if first_region.height < (self.predicted_line_height * MULT_1):
            self.avg_line_height += first_region.height

        self.lines_region = [first_region]

        # Add the rest of regions.
        for i in range(len(self.initial_lines)):
            top_line = self.initial_lines[i]

            if i + 1 < len(self.initial_lines):
                bottom_line = self.initial_lines[i + 1]
            else:
                bottom_line = Line()

            next_region = Region(top_line, bottom_line)
            is_all_white = next_region.update_region(self.gray_img, i, debug)
            
            if top_line.initial_valley_id != -1:
                top_line.below = next_region

            if bottom_line.initial_valley_id != -1:
                bottom_line.above = next_region

            if is_all_white is False:
                self.lines_region.append(next_region)

                if (next_region.height < self.predicted_line_height * MULT_1):
                    self.avg_line_height += next_region.height

        if len(self.lines_region) > 0:
            self.avg_line_height /= len(self.lines_region)

    def component_belongs_to_above_region(self, line, contour):
        # Calculate probabilities
        probAbovePrimes = [0] * len(self.primes)
        probBelowPrimes = [0] * len(self.primes)
        n = 0

        tl = (contour[0], contour[1])  # top left

        width, height = contour[2], contour[3]

        for i_contour in range(tl[0], tl[0] + width):
            for j_contour in range(tl[1], tl[1] + height):
                if self.thresh[j_contour][i_contour] == 255:
                    continue

                n += 1

                contour_point = np.zeros([1, 2], dtype=np.uint8)
                contour_point[0][0] = j_contour
                contour_point[0][1] = i_contour
                
                if line.above != 0:
                    mean, cov = line.above.mean, line.above.covariance
                    newProbAbove = Region.bi_variate_gaussian_density(contour_point, mean, cov)
                else:
                    newProbAbove = 0

                if line.below != 0:
                    mean, cov = line.below.mean, line.below.covariance
                    newProbBelow = Region.bi_variate_gaussian_density(contour_point, mean, cov)
                else:
                    newProbBelow = 0

                LineSegmentation.add_primes_to_list(self.primes, newProbAbove, probAbovePrimes)
                LineSegmentation.add_primes_to_list(self.primes, newProbBelow, probBelowPrimes)

        prob_above = 0
        prob_below = 0

        for k in range(len(probAbovePrimes)):
            mini = min(probAbovePrimes[k], probBelowPrimes[k])

            probAbovePrimes[k] -= mini
            probBelowPrimes[k] -= mini

            prob_above += probAbovePrimes[k] * self.primes[k]
            prob_below += probBelowPrimes[k] * self.primes[k]


        return prob_above < prob_below, line, contour

    def repair_lines(self):
        """
        repeair all initial lines and generate the final line region
        """

        for line in self.initial_lines:
            column_processed = {}  # int, bool

            for column in range(self.img.shape[1]):#cols
                column_processed[column] = False

            i = 0
            while i < len(line.points):
                point = line.points[i]
                x = int(point[0])
                y = int(point[1])
                # print(y)
                # Check for vertical line intersection
                # In lines, we don't save all the vertical points we save only the start point and the end point.
                # So in line->points all we save is the horizontal points so, we know there exists a vertical line by
                # comparing the point[i].x (row) with point[i-1].x (row)
                if self.thresh[x][y] == 255:
                    if i == 0:
                        i+=1
                        continue
                    black_found = False
                    if line.points[i - 1][0] != line.points[i][0]:
                        # Means the points are in different rows (a vertical line).
                        min_row = int(min(line.points[i - 1][0], line.points[i][0]))
                        max_row = int(max(line.points[i - 1][0], line.points[i][0]))

                        for j in range(int(min_row), int(max_row) + 1):
                            if black_found is True:
                                break
                            if self.thresh[j][line.points[i - 1][1]] == 0:
                                x = j
                                y = line.points[i - 1][1]
                                black_found = True

                    if black_found == False:
                        i+=1
                        continue

                # Ignore it's previously processed

                if column_processed[y] == True:
                    i+=1
                    continue

                # Mark column as processed
                column_processed[y] = True

                self.avg_line_height = int(self.avg_line_height)

                for c in self.contours:
                    # Check line & contour intersection
                    tl = (c[0], c[1])
                    br = (c[0] + c[2], c[1] + c[3])

                    if y >= tl[0] and y <= br[0] and x >= tl[1] and x <= br[1]:
                        # print("br: {}".format(br))
                        # If contour is longer than the average height ignore
                        height = br[1] - tl[1]

                        if height > int(self.avg_line_height * 0.9):
                            continue

                        is_component_above, line, c = self.component_belongs_to_above_region(line, c)

                        # print(is_component_above)
                        new_row = 0
                        if is_component_above == False:
                            new_row = tl[1]
                            line.min_row_pos = min(line.min_row_pos, new_row)
                        else:
                            new_row = br[1]
                            line.max_row_pos = max(new_row, line.max_row_pos)

                        width = c[2]

                        for k in range(tl[0], tl[0] + width):
                            point_new = (new_row, line.points[k][1]) # make this coz tuple does not have value-assignment
                            if k < len(line.points):
                                line.points[k] = point_new
                            else:
                                self.logger.error(f"Can't save a new point at pos {k} if length is {len(line.points)}")

                        i = br[0]  # bottom right

                        # print("I: {}".format(i))
                        break  # Contour found
                i += 1

    def get_regions(self, strip_whites: bool, image_width: int, buffer_lines: int = 10, whitest_range: int = 10):
        """If 'strip_whites' is set to True, all empty space from a line image will be removed"""
        lines: NumpyArray = [region.region for region in self.lines_region]
        coords: NumpyArray = [(region.start, region.end) for region in self.lines_region]

        if strip_whites:
            if not issubclass(lines[0].dtype.type, Integral):
                raise Exception("Expected line images to be in the range 0..255")

            new_lines, new_coords = [], []

            for line_image, (start, end) in zip(lines, coords):
                hproj = np.sum((line_image / 255), axis=1).astype(np.int32)  # horizontal projection
                text_hlines = np.where(hproj < (image_width - whitest_range))[0]

                stripped_line, stripped_coord = self._resize_line(line_image, text_hlines, buffer_lines, start, end)

                new_lines.append(stripped_line)
                new_coords.append(stripped_coord)

            lines = new_lines
            coords = new_coords

        return lines, np.array(coords)
                
    
    def _resize_line(self, line_image, text_hlines, num_buffer_lines, start, end):
        """"""
        lower_limit, upper_limit = text_hlines[0] - num_buffer_lines, text_hlines[-1] + num_buffer_lines
        image_height, _ = self.img.shape[:2]
        line_height, _ = line_image.shape[:2]

        # start, end are the start & end of the line w.r.t. the image
        # text_hlines[0], text_hlines[-1] are the start & end of the line w.r.t. itself

        new_start, new_end = start, end
        pre_whites, post_whites = [], []

        if start == 0:
            new_end = text_hlines[-1] + num_buffer_lines
            post_whites = np.arange(text_hlines[-1] + 1, new_end)

        elif end == image_height:
            new_start = text_hlines[0] - num_buffer_lines
            pre_whites = np.arange(new_start, text_hlines[0] - 1)

        elif (lower_limit < 0) or (upper_limit > image_height):
            raise Exception(f"New limits {lower_limit}, {upper_limit} are not in the image range [0..{image_height}]")
        elif (upper_limit > line_height):
            raise Exception(f"New limits {lower_limit}, {upper_limit} are not in the line range [0..{line_height}]")
        else:
            new_start = text_hlines[0] - num_buffer_lines
            new_end = text_hlines[-1] + num_buffer_lines

            pre_whites = np.arange(new_start, text_hlines[0] - 1)
            post_whites = np.arange(text_hlines[-1] + 1, new_end)
        
        pos = np.array([*pre_whites, *text_hlines, *post_whites])
        return line_image[pos, :], (new_start, new_end)

    def save_image_with_lines(self, path):
        """"""
        img_clone = self.img.copy()

        for line in self.initial_lines:
            last_row = -1

            for point in line.points:
                img_clone[int(point[0])][int(point[1])] = (0, 0, 255)
                # Check and draw vertical lines if found.
                if last_row != -1 and point[0] != last_row:
                    for i in range(min(int(last_row), int(point[0])), max(int(last_row), int(point[0]))):
                        img_clone[int(i)][int(point[1])] = (0, 0, 255)

                last_row = point[0]

        # cv2.imwrite(path, img_clone)

    def save_lines_to_file(self, lines):
        """
        lines: list contains multiple images as numpy arrays
        Return pathes containing various output image pathes
        """
        output_image_path = []
        idx = 0
        if len(self.initial_lines) == 0:
            path = self.output_path + "/Line_" + str(idx) + ".jpg"
            output_image_path.append(path)
            cv2.imwrite(path, self.img)
            return output_image_path

        idx = 0
        output_image_path = []
        for m in lines:
            path = self.output_path + "/Line_" + str(idx) + ".jpg"
            cv2.imwrite(path, m)
            output_image_path.append(path)
            idx += 1

        return output_image_path
