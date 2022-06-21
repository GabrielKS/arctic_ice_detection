import numpy as np
import cv2

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
