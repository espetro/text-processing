
from numpy.random import randint

class ParagraphSegmentation:
    """Breaks a given text image into paragraphs, following the peak-valley model.
    This class requires a LineSegmentation instance to be ran previously, in order to obtain the valley lines
    that split text lines (or "peaks").
    """

    def __init__(self, image, valleys):
        self.image = image
        self.vall = valleys

    def get_paragraphs(self):
        """Returns the paragraphs groups as a list of lines, where each line is a range(valley_down, valley_up) from
        the image"""
        pass

    def get_peaks(self):
        """Obtains the set of peaks (text lines) in an image"""
        pass

    def find_peak_between(self, v_down, v_up, jump_dist=1):
        """Finds a peak given two valleys. Intuitively, the peak shall be in the exact middle between 2 valleys. If not,
        start searching it from the middle to the valleys.
        
        If no peak is found, a RuntimeError is raised"""
        if jump_dist < 0:
            raise ValueError("Jump distance must be greater than 0")

        curr_line = start = (v_down + v_up) // 2
        next_up = (x for x in range(start - jump_dist, v_down, -jump_dist))
        next_bot = (x for x in range(start + jump_dist, v_up, jump_dist))

        if not is_peak(curr_line):
            while line_up != -1 or line_down != -1:
                line_up = next(next_up, -1)
                line_down = next(next_bot, -1)

                if line_up != -1 and is_peak(line_up):
                    return line_up
                elif line_down != -1 and is_peak(line_down):
                    return line_down

            raise RuntimeError(f"Expected to find a peak between ({v_down}, {v_up}) but none was found")
        else:
            return curr_line

    @staticmethod
    def is_peak(arr):
        """Checks if the given line belongs to a peak (sum > 0)"""
        return int(np.sum(arr)) > 0

    @staticmethod
    def is_valley(arr):
        """Checks if the given line belongs to a valley (sum == 0)"""
        return int(np.sum(arr)) == 0
