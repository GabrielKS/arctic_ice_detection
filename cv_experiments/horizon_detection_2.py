"""The Hough transform applied to more images"""

import os
import cv2
import cv_experiments.cv_common as cc
import numpy as np

input_dir = os.path.abspath("../representatives/all")

def canny(img):
    edges = img
    edges = cv2.resize(edges, (120, 67))
    edges = cv2.Canny(edges, 12800, 22400, apertureSize=7)
    return edges

def hough(img, dest, line_weight=3):
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 50, None, 5, 15)
    xscale = dest.shape[1]/img.shape[1]
    yscale = dest.shape[0]/img.shape[0]

    if lines is not None:
        # Find the longest line and extend it across the frame
        (x1, y1, x2, y2) = max(lines, key=lambda line: (line[0][2]-line[0][0])**2+(line[0][1]-line[0][3])**2)[0]
        slope = (y2-y1)/(x2-x1)
        if abs(slope) < 100:  # Basic sanity check
            longest_line_extended = (0, int((y1-slope*x1)*yscale)), (dest.shape[1], int((y2+slope*(img.shape[1]-x2))*yscale))
            cv2.line(dest, *longest_line_extended, (0, 255, 0), line_weight, cv2.LINE_AA)

        for line in lines:
            cv2.line(dest, (int(line[0][0]*xscale), int(line[0][1]*yscale)),
                (int(line[0][2]*xscale), int(line[0][3]*yscale)),
                (0, 0, 255), line_weight, cv2.LINE_AA)
    return dest

def annotate_image(img):
    orig_size = np.array([img.shape[1], img.shape[0]])
    edges = canny(img)
    lines = hough(edges, img)
    combo = cv2.addWeighted(lines, 1, cv2.cvtColor(cv2.resize(edges, orig_size), cv2.COLOR_GRAY2BGR), 0.25, 0)
    return combo

def main():
    images = cc.load_dir(input_dir, colormode=cv2.IMREAD_COLOR)
    processed = [cv2.resize(annotate_image(img), (650, 350)) for img,_ in images]
    cv2.imshow("hough", cc.tile(processed, 1650*2, 950*2))
    for _ in range(2): cv2.moveWindow("hough", -1800, 0)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
