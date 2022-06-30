import numpy as np
import cv2
import os

focal_length = 10
distCoeffs = {
        "small": np.array([-3.31e-4, 1.24e-8, 0, 0]),
        "large": np.array([-4.18e-5,  1.78e-9, 0, 0])
    }

def undistort(img, cam_id):
    cam = np.array([
        [focal_length,  0,              img.shape[1]/2],
        [0,             focal_length,   img.shape[0]/2],
        [0,             0,              1]])
    return cv2.undistort(img, cam, distCoeffs[cam_id])

def imscale(img, scale):
    return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

def get_image(filename, colormode=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(filename, colormode)
    return undistort(img, "large" if img.shape[0] > 1000 else "small")

def load_dir(input_dir, colormode=cv2.IMREAD_GRAYSCALE):
    image_paths = sorted(f.path for f in os.scandir(input_dir) if f.is_file() and os.path.basename(f.path)[0] != '.')
    sources = list(zip(map(lambda f: get_image(f, colormode=colormode), image_paths), map(lambda f: os.path.basename(f), image_paths)))
    return sources

def tile(images, twidth, theight):
    out = np.zeros((theight, twidth, 3), dtype=np.uint8)
    iheight, iwidth = images[0].shape[:2]
    cols = twidth//iwidth
    for i, img in enumerate(images):
        row, col = i//cols, i%cols
        out[iheight*row:iheight*(row+1), iwidth*col:iwidth*(col+1), :] = img
    return out
