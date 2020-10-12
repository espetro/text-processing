from numpy import ndarray

import numpy as np
import numba as nb
import cv2

class IlluminationBinarizer:
    """
    An implementation of Binarize with a robust version of Illumination Compensation.

    Illumination Compensation based in:
        K.-N. Chen, C.-H. Chen, C.-C. Chang,
        Efficient illumination compensation techniques for text images, in
        Digital Signal Processing, 22(5), pp. 726-733, 2012.
    
    For a shorter version of the algorithm, see on GitHub:
        @fanyirobin/text-image-binarization
        @arthurflor23/handwritten-text-recognition

    This class uses numba to accelerate numpy-based functions
    """

    @staticmethod
    def imread(fpath: str, grayscale=False):
        """Reads an image in grayscale mode"""
        if grayscale:
            return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_grayscale(image: ndarray, from_rgb=True):
        """Transforms an image from RGB to Grayscale"""
        if from_rgb:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def illumination_compensation(image: ndarray, as_bin=False, upper_value=255, c:float=0.3, bl:int=260, iters=1,
        debug=False):
        """Computes the illumination-compensated image for the given image.
        
        Parameters
        ----------
            image: 3D numpy array
                A RGB image
            c: int, default 0.3
                A parameter in the interval [0.1..0.4] used to further reduce image brightness.
            bl: int, default 260.
                Recommended values: [200..300]
                                
            iters: int, default 1.
                Number of iterations to run. Only works if debug is set to False.

            debug: bool, default False
                If True, runs Numba-compiled as usual Python functions
        Returns
        -------
            2D numpy array, the binarized result image
        """
        if not isinstance(iters, int) or iters < 1:
            raise ValueError("The number of iterations must be greater than 1")
        
        gray = IlluminationBinarizer.to_grayscale(image, from_rgb=True).astype(np.float32)
        height, width = gray.shape[:2]

        if debug:
            cei = IlluminationBinarizer.enhance_contrast.py_func(image, gray, c)
            edges = IlluminationBinarizer.detect_edges(gray)
            erosion = IlluminationBinarizer.locate_text(cei, edges)
            compute_mat = np.array(cei)
            IlluminationBinarizer.estimate_lightness_distr.py_func(compute_mat, height, width, erosion, cei)
            mean_filter = 1 / 121 * np.ones((11,11), np.uint8)
            ldi = cv2.filter2D(IlluminationBinarizer.minmax_scale.py_func(compute_mat), -1, mean_filter)
            result = IlluminationBinarizer.compute_result.py_func(cei, erosion, ldi, bl).astype(np.uint8)

            if as_bin:
                thresh = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                return IlluminationBinarizer.minmax_scale.py_func(thresh, upper_value).astype(np.uint8)
            else:
                return result
        else:
            result = image
            for _ in range(iters):
                # 1. Enhance contrast
                cei = IlluminationBinarizer.enhance_contrast(result, gray, c)

                # 2. Edge detection
                edges = IlluminationBinarizer.detect_edges(gray)

                # 3. Locate text
                erosion = IlluminationBinarizer.locate_text(cei, edges)

                # 4. Estimate lightness distribution
                compute_mat = np.array(cei)
                IlluminationBinarizer.estimate_lightness_distr(compute_mat, height, width, erosion, cei)

                mean_filter = 1 / 121 * np.ones((11,11), np.uint8)
                ldi = cv2.filter2D(
                    IlluminationBinarizer.minmax_scale(compute_mat), -1, mean_filter
                )

                # 5. Compute result
                result = IlluminationBinarizer.compute_result(cei, erosion, ldi, bl).astype(np.uint8)

            if as_bin:
                thresh = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                return IlluminationBinarizer.minmax_scale(thresh, upper_value).astype(np.uint8)
            else:
                return result

    @staticmethod
    @nb.njit(cache=True)
    def enhance_contrast(image: ndarray, gray: ndarray, c:float):
        height, width = gray.shape[:2]

        bins = np.arange(0, 300, 10)
        bins[26] = 255
        hist = np.histogram(image, bins)
        
        sqrt_hw, hist_dim, hr = (np.sqrt(height * width), len(hist[0]), 0)
        for i in range(hist_dim):
            if hist[0][i] > sqrt_hw:
                hr = i * 10
                break

        cei = (gray - (hr + 50 * c)) * 2
        for x in range(height):
            for y in range(width):
                if cei[x, y] > 255:
                    cei[x, y] = 255
                elif cei[x, y] < 0:
                    cei[x, y] = 0
        
        return cei

    @staticmethod
    def detect_edges(gray: ndarray):
        """Applies edge detection"""
        m1 = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3))
        m2 = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape((3,3))
        m3 = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape((3,3))
        m4 = np.array([0,1,2,-1,0,1,-2,-1,0]).reshape((3,3))

        eg1 = np.abs(cv2.filter2D(gray, -1, m1))
        eg2 = np.abs(cv2.filter2D(gray, -1, m2))
        eg3 = np.abs(cv2.filter2D(gray, -1, m3))
        eg4 = np.abs(cv2.filter2D(gray, -1, m4))
        edge_avg = IlluminationBinarizer.minmax_scale((eg1 + eg2 + eg3 + eg4) / 4)

        return edge_avg

    @staticmethod
    @nb.njit(cache=True)
    def minmax_scale(image: ndarray, upper_value=255):
        """Rescales an image to range 0..upper_value"""
        s = np.max(image) - np.min(image) 
        res = image / s
        res -= np.min(res)
        res *= upper_value
        return res

    @staticmethod
    @nb.njit(cache=True)
    def apply_threshold(image: ndarray, th:int, flag:int):
        """Applies a threshold to the image.
        If 0, then 'image < th' applies, if 1 then 'image >= th' applies.
        """
        height, width = image.shape[:2]
        new_image = np.zeros((height, width))

        if flag == 0:
            for x in range(height):
                for y in range(width):
                    if image[x, y] < th:
                        new_image[x, y] = 255
        elif flag == 1:
            for x in range(height):
                for y in range(width):
                    if image[x, y] >= th:
                        new_image[x, y] = 255
        return new_image

    @staticmethod
    @nb.njit(cache=True)
    def merge(image1: ndarray, image2: ndarray):
        """Both images must have the same dimensions"""
        height, width = image1.shape[:2]
        merged_image = np.ones((height, width)) * 255

        for x in range(height):
            for y in range(width):
                if image1[x, y] == 255:
                    merged_image[x, y] = 0
                if image2[x, y] == 255:
                    merged_image[x, y] = 0

        return merged_image

    @staticmethod
    def locate_text(cei: ndarray, edges: ndarray, th1:int=30, th2:int=60):
        """Locate text within image.
        Thresholds are hard-coded (based in the paper)."""
        edge_bin = IlluminationBinarizer.apply_threshold(edges, th1, flag=1)
        cei_bin = IlluminationBinarizer.apply_threshold(cei, th2, flag=0)

        tli = IlluminationBinarizer.merge(edge_bin, cei_bin)
        kernel = np.ones((3,3), np.uint8)
        return cv2.erode(tli, kernel, iterations=1)
        # compute_mat = np.asarray(cei)

    @staticmethod
    @nb.njit(cache=True)
    def estimate_lightness_distr(compute_mat: ndarray, height:int, width:int, erosion: ndarray, cei: ndarray):
        for y in range(width):
            for x in range(height):
                if erosion[x, y] == 0:
                    # set_intp_img(compute_mat, x, y, erosion, cei)
                    head = x
                    
                    # find_end(erosion, x, y)
                    i = x  # to loop over erosion
                    while (i < erosion.shape[0]) and (erosion[i, y] == 0):
                        i += 1
                    end = i - 1
                    n = end - head + 1

                    if n <= 30:
                        # find_mpv(cei, head, end, y)
                        h = np.zeros(5, dtype=cei.dtype)
                        e = np.zeros(5, dtype=cei.dtype)

                        for k in range(5):
                            if head - k >= 0:
                                h[k] = cei[(head - k), y]
                            if end + k < cei.shape[0]:
                                e[k] = cei[(end + k), y]

                        mpv_h, mpv_e = np.max(h), np.max(e)

                        for m in range(n):
                            new_x = head + m
                            compute_mat[new_x, y] = mpv_h + (m + 1) * ((mpv_e - mpv_h) / n) 

                    x = end  # this is done for both (n > 30, n <= 30)
                #endif
            #endfor
        #endfor

    @staticmethod
    @nb.njit(cache=True)
    def compute_result(cei: ndarray, erosion: ndarray, ldi: ndarray, bl: int):
        result = np.divide(cei, ldi) * bl
        height, width = result.shape[:2]

        for x in range(height):
            for y in range(width):
                if erosion[x, y] != 0:
                    result[x, y] *= 1.5

                if result[x, y] < 0:
                    result[x, y] = 0
                elif result[x, y] > 255:
                    result[x, y] = 255

        return result


    @staticmethod
    def _calc_y_alpha(vec):
        """Util function used in 'remove_cursing_style'"""
        indices = np.where(vec > 0)[0]
        h_alpha = len(indices)

        if h_alpha > 0:
            delta_y_alpha = indices[h_alpha - 1] - indices[0] + 1

            if h_alpha == delta_y_alpha:
                return h_alpha * h_alpha
        return 0

    @staticmethod
    def remove_cursive_style(img):
        """Remove cursive writing style from image with deslanting algorithm.
        
        Deslating image process based in,
            A. Vinciarelli and J. Luettin,
            A New Normalization Technique for Cursive Handwritten Wrods, in
            Pattern Recognition, 22, 2001.
        """
        # this algorithm needs a binarized image
        img = (img > 100).astype(np.uint8)
        # img = IlluminationBinarizer.illumination_compensation(img) 

        alpha_vals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        rows, cols = img.shape
        results = []

        ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = otsu if ret < 127 else sauvola(img, (int(img.shape[0] / 2), int(img.shape[0] / 2)), 127, 1e-2)

        for alpha in alpha_vals:
            shift_x = max(-alpha * rows, 0.)
            size = (cols + int(np.ceil(abs(alpha * rows))), rows)
            transform = np.asarray([[1, alpha, shift_x], [0, 1, 0]], dtype=np.float)

            shear_img = cv2.warpAffine(binary, transform, size, cv2.INTER_NEAREST)
            sum_alpha = 0
            sum_alpha += np.apply_along_axis(IlluminationBinarizer._calc_y_alpha, 0, shear_img)
            results.append([np.sum(sum_alpha), size, transform])

        result = sorted(results, key=lambda x: x[0], reverse=True)[0]
        warp = cv2.warpAffine(img, result[2], result[1], borderValue=255)

        return cv2.resize(warp, dsize=(cols, rows))