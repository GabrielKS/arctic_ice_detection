"""
A quick and dirty test of the hypothesis that if we undistort an image, rotate it, blur horizontally across the entire
image (i.e., basically take an average), and take a vertical cross section, the standard deviation of that cross
section will be maximized if the rotation angle is such that the horizon is perfectly horizontal. With one of my test
images, in which the sky and ocean are clearly different colors, both are pretty uniform on the macroscale, and the
horizon line is sharp, this works perfectly. With another test image, where clouds make the sky more complicated, there
is a local maximum there, but not an absolute maximum.
"""

import os
import numpy as np
import cv2
from . import cv_common as cc

input_dir = os.path.abspath("../representatives/rotation")
transformed_suffix = " blurred"
sources = None

def get_images():
    global sources
    sources = [(cc.imscale(img, 0.58) if img.shape[0] > 1000 else img, name) for img, name in cc.load_dir(input_dir)]

def mouser(event, x, y, flags, param):
    angle = (x-200)/400*(np.pi/2)
    for i, (img, name) in enumerate(sources):
        process_image(img, name, angle)
    print()

def rotate_rect_dims(width, height, angle):  # Inspired by https://stackoverflow.com/a/16778797
    longer, shorter = max(width, height), min(width, height)
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if shorter <= 2*sin_a*cos_a*longer or abs(sin_a-cos_a) < 1e-5:
        rlonger, rshorter = 0.5*shorter/sin_a, 0.5*shorter/cos_a
        rwidth, rheight = (rlonger, rshorter) if width > height else (rshorter, rlonger)
    else:
        csd = cos_a*cos_a-sin_a*sin_a
        rwidth, rheight = (width*cos_a - height*sin_a)/csd, (height*cos_a - width*sin_a)/csd
    return rwidth, rheight

def rotate_crop(img, angle):
    height, width = img.shape[:2]
    sf = min(width, height)/np.sqrt(width**2+height**2)
    img = cc.imscale(img, 1/sf)
    height, width = img.shape[:2]
    rwidth, rheight = rotate_rect_dims(width, height, angle)
    m = cv2.getRotationMatrix2D((width//2, height//2), angle*180/np.pi, sf)
    rotated = cv2.warpAffine(img, m, (width, height))
    xo = int((width-rwidth*sf)/2)+5
    yo = int((height-rheight*sf)/2)+5
    rotated = rotated[yo:rotated.shape[0]-yo, xo:rotated.shape[1]-xo]
    return rotated

def mount(rotated, owidth, oheight):
    e = np.ones((oheight, owidth), np.uint8)*127
    xo = int((owidth-rotated.shape[1])/2)
    yo = int((oheight-rotated.shape[0])/2)
    e[yo:rotated.shape[0]+yo, xo:rotated.shape[1]+xo] = rotated
    return e

def transform(img, name):
    safe_height = 100
    out = cv2.blur(img, (10000, 1))  # Blur across the entire image horizontally, none vertically
    column = out[:, 0]
    height = len(column)
    # Crop to a uniform height
    column = column[int(height/2-safe_height/2):int(height/2+safe_height/2)]
    print(f"{name}: {column.std():.2f}")
    return out

def process_image(img, name, angle):
    height, width = img.shape[:2]
    rotated = rotate_crop(img, angle)
    transformed = transform(rotated, name)

    owidth, oheight = [max(width, height)]*2
    out = mount(rotated, owidth, oheight)
    cv2.imshow(name, out)
    out2 = mount(transformed, owidth, oheight)
    cv2.imshow(name+transformed_suffix, out2)

def main():
    get_images()
    for i, (img, name) in enumerate(sources):
        process_image(img, name, 0)
        for _ in range(2): cv2.moveWindow(name, -1500+600*i, 0)
        for _ in range(2): cv2.moveWindow(name+transformed_suffix, -1500+600*i, 600)
    for _,name in sources:
        cv2.setMouseCallback(name, mouser)
        cv2.setMouseCallback(name+transformed_suffix, mouser)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
