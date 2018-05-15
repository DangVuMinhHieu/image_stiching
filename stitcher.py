import cv2
import numpy as np
import argparse
from matcher import Matcher


class Stitcher(object):
    def __init__(self):
        self.my_matcher = Matcher()

    def calculated_dsize(self, img1, img2, homo):
        top_left = np.dot(homo, np.array([0, 0, 1]))
        top_left = top_left / top_left[-1]
        down_right = np.dot(homo, np.array([img2.shape[1], img2.shape[0], 1]))
        down_right = down_right / down_right[-1]
        top_right = np.dot(homo, np.array([img2.shape[1], 0, 1]))
        top_right = top_right / top_right[-1]
        down_left = np.dot(homo, np.array([0, img2.shape[0], 1]))
        down_left = down_left / down_left[-1]
        y_max = max([top_left[1], top_right[1], down_right[1], down_left[1]])
        y_min = min([top_left[1], top_right[1], down_right[1], down_left[1]])
        (row_left, col_left) = img1.shape[:2]
        d_size = (np.float32(img2.shape[1] + col_left), np.float32(max(abs(y_max - y_min), row_left)))
        return d_size

    def stitch_two_images(self, image_left, image_right, homo):
        (row_left, col_left) = image_left.shape[:2]
        d_size = self.calculated_dsize(image_left, image_right, homo)
        img = cv2.warpPerspective(image_right, homo, d_size)
        img[0: row_left, 0: col_left] = np.maximum(img[0: row_left, 0: col_left], image_left)
        return img

    def stitch_multiple_images(self, images, use_opencv_homo=False):
        anchor = images[0]
        for image in images[1:]:
            if use_opencv_homo:
                homo = self.my_matcher.feature_matching_using_opencv(
                    image, anchor, show_result=False)
            else:
                homo = self.my_matcher.feature_matching(image, anchor, show_result=False)
            anchor = self.stitch_two_images(anchor, image, homo)
        return anchor

    def show_result_image(self, img, name="result"):
        self.my_matcher.show_result_image(img, name)
