import numpy as np
import cv2
import os

focal_length = 10
distCoeffs = {  # Coefficients and the image resolutions at which they were determined, plus optionally scale factor
        "small": (np.array([-3.31e-4, 1.24e-8, 0, 0]), np.array([1072, 1920])),
        "large": (np.array([-4.18e-5,  1.78e-9, 0, 0]), np.array([279, 500])),
        "new": (np.array([-5.35e-4, 3.05e-7, 0, 0]), np.array([279, 500]), 0.7845)
    }

def undistort(img, cam_id, interpolation_method=cv2.INTER_LINEAR, fill_value=0):
    # We can compensate for the image having been resized by adjusting the focal length
    fl_scale = (img.shape[:2]/distCoeffs[cam_id][1]).mean()  # Assumes resizing has occurred ~equally in both dimensions
    cam = np.array([
        [focal_length*fl_scale,  0,              img.shape[1]/2],
        [0,             focal_length*fl_scale,   img.shape[0]/2],
        [0,             0,              1]])
    # return cv2.undistort(img, cam, distCoeffs[cam_id][0])
    # Equivalent but lets us set the interpolation method ourselves:
    map1, map2 = cv2.initUndistortRectifyMap(cam, distCoeffs[cam_id][0], np.eye(3), cam,
        (img.shape[1], img.shape[0]), cv2.CV_32FC1)
    if len(distCoeffs[cam_id]) >= 3:
        img_scale = distCoeffs[cam_id][2]
        map1 = map1/img_scale-img.shape[1]*(1/img_scale-1)/2
        map2 = map2/img_scale-img.shape[0]*(1/img_scale-1)/2
    return cv2.remap(img, map1, map2, interpolation=interpolation_method, borderValue=fill_value)

def imscale(img, scale):
    return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

def get_image(filename, colormode=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(filename, colormode)
    return undistort(img, "large" if img.shape[0] > 1000 else "small")

def load_dir(input_dir, colormode=cv2.IMREAD_GRAYSCALE):  # It's lazy -- only actually loads images when they're called for
    image_paths = sorted(f.path for f in os.scandir(input_dir) if f.is_file() and os.path.basename(f.path)[0] != '.')
    sources = zip(map(lambda f: get_image(f, colormode=colormode), image_paths), map(lambda f: os.path.basename(f), image_paths))
    return sources

def tile(images, twidth, theight):
    out = np.zeros((theight, twidth, 3), dtype=np.uint8)
    iheight, iwidth = images[0].shape[:2]
    cols = twidth//iwidth
    for i, img in enumerate(images):
        row, col = i//cols, i%cols
        out[iheight*row:iheight*(row+1), iwidth*col:iwidth*(col+1), :] = img
    return out
