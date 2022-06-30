"""
Like interactive_undistorter.py, but with more images and with automatic horizon detection feedback.
Didn't end up being that helpful.
"""

import os
import numpy as np
import cv2
import cv_experiments.cv_common as cc
from cv_experiments.horizon_detection_2 import annotate_image, hough, canny

input_dir = os.path.abspath("../representatives/better_undistort")

xrange, yrange = 6, 6  # How many powers of ten to move across in each direction
xoffset, yoffset = -10, -10  # Minimum power of ten we care about
threshold_image = False  # Whether to apply some thresholding to the image
focal_length = 10
dwidth, dheight = 700, 400  # Display width and height

inputs = {"sources": None, "names": None, "width": None, "height": None}

def get_images():
    inputs["sources"], inputs["names"] = zip(*cc.load_dir(input_dir, colormode=cv2.IMREAD_COLOR))
    inputs["width"] = inputs["sources"][0].shape[1]
    inputs["height"] = inputs["sources"][0].shape[0]

    for this_src in inputs["sources"]:
        if threshold_image:
            this_src = cv2.GaussianBlur(this_src, (15, 15), 0)
            this_src = cv2.adaptiveThreshold(this_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, inputs["height"]//2*2+1, 0)

def undistort(x, y):
    # Exponential with sign added: greater towards edges, close to zero at center, down+right is positive
    coeff_1  = np.sign(x-dwidth/2)*10**(np.abs((x-dwidth/2)/(dwidth/2)*xrange)+xoffset)
    coeff_2 = np.sign(y-dheight/2)*10**(np.abs((y-dheight/2)/(dheight/2)*yrange)+yoffset)
    print(f"{coeff_1:.3E}, {coeff_2:.3E}")  # Write these down when you find a good undistortion

    distCoeffs = np.array([coeff_1, coeff_2, 0, 0])
    cam = np.array([
        [focal_length,  0,              inputs["width"]/2],
        [0,             focal_length,   inputs["height"]/2],
        [0,             0,              1]])
    
    for img,name in zip(inputs["sources"], inputs["names"]):
        dst = cv2.undistort(img, cam, distCoeffs)
        dst = hough(canny(dst), dst, line_weight=1)
        dst = cv2.resize(dst, (dwidth, dheight))
        cv2.imshow(name, dst)

def mouser(event, x, y, flags, param):
    undistort(x, y)

def main():
    get_images()
    for name in inputs["names"]:
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, mouser)
    undistort(dwidth/2, dheight/2)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
