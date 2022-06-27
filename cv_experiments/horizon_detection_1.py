"""A first attempt at detecting the horizon using Canny edge detection and the probabilistic Hough line transform"""

import os
import cv2
import cv_experiments.cv_common as cc
import numpy as np

input = os.path.abspath("../representatives/hough_example_hard.jpg")

def canny(img):
    edges = img
    # Scale down to reduce the impact of the horizon being blurry
    edges = cv2.resize(edges, (120, 67))
    # Values tuned experimentally
    edges = cv2.Canny(edges, 12800, 22400, apertureSize=7)
    return edges

def hough(img, dest):
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 50, None, 2, 5)
    xscale = dest.shape[1]/img.shape[1]
    yscale = dest.shape[0]/img.shape[0]
    for line in lines:
        cv2.line(dest, (int(line[0][0]*xscale), int(line[0][1]*yscale)),
            (int(line[0][2]*xscale), int(line[0][3]*yscale)),
            (0, 0, 255), 3, cv2.LINE_AA)
    return dest

def main():
    img = cv2.imread(input)
    img = cc.undistort(img, "large")
    orig_size = np.array([img.shape[1], img.shape[0]])
    edges = canny(img)
    lines = hough(edges, img)
    combo = cv2.addWeighted(lines, 1, cv2.cvtColor(cv2.resize(edges, orig_size), cv2.COLOR_GRAY2BGR), 0.25, 0)
    cv2.imshow("hough", combo)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
