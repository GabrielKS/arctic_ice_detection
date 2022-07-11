# Like the original but with two windows, one for images from each camera

import os
import numpy as np
import cv2

input_dir = os.path.abspath("../representatives/undistort_3")
window_name = "Interactive Undistorter"
cameras = ["2", "3"]

# Settings that may need to be adjusted depending on the input:
# crop = slice(None)  # No cropping
crop = [slice(100, -100), slice(200, -200)]  # Some cropping
xrange, yrange = 6, 6  # How many powers of ten to move across in each direction
xoffset, yoffset = -8, -8  # Minimum power of ten we care about
threshold_image = False  # Whether to apply some thresholding to the image
focal_length = 10

inputs = {"sources2": None, "sources3": None, "combo2": None, "combo3": None, "width": None, "height": None}

def get_images(cam_id):
    inputs[f"sources{cam_id}"] = [cv2.imread(os.path.join(input_dir, input_img), cv2.IMREAD_GRAYSCALE)
        for input_img in os.listdir(input_dir) if input_img[-5:] == f"{cam_id}.png"]
    
    # Assumes all images are the same size
    inputs["width"] = inputs[f"sources{cam_id}"][0].shape[1]
    inputs["height"] = inputs[f"sources{cam_id}"][0].shape[0]
    print(inputs["width"])
    print(inputs["height"])

    inputs[f"combo{cam_id}"] = np.zeros(inputs[f"sources{cam_id}"][0].shape, np.float32)
    n_inputs = len(inputs[f"sources{cam_id}"])
    for this_src in inputs[f"sources{cam_id}"]:
        inputs[f"combo{cam_id}"] += (this_src/n_inputs)
    inputs[f"combo{cam_id}"] = inputs[f"combo{cam_id}"].astype(np.uint8)

def undistort(x, y):
    # Exponential with sign added: greater towards edges, close to zero at center, down+right is positive
    coeff_1  = np.sign(x-inputs["width"]/2)*10**(np.abs((x-inputs["width"]/2)/(inputs["width"]/2)*xrange)+xoffset)
    coeff_2 = np.sign(y-inputs["height"]/2)*10**(np.abs((y-inputs["height"]/2)/(inputs["height"]/2)*yrange)+yoffset)
    print(f"{coeff_1:.2E}, {coeff_2:.2E}")  # Write these down when you find a good undistortion

    # Experimental evidence suggests:
    # Parameter 1: radially symmetric, affects the outer edges more
    # Parameter 2: radially symmetric, affects the outer edges way more
    # Parameter 3: pitch
    # Parameter 4: yaw
    distCoeffs = np.array([coeff_1, coeff_2, 0, 0])
    cam_mat = np.array([
        [focal_length,  0,              inputs["width"]/2],
        [0,             focal_length,   inputs["height"]/2],
        [0,             0,              1]])
    
    for cam_id in cameras:
        dst = cv2.undistort(inputs[f"combo{cam_id}"], cam_mat, distCoeffs)
        cv2.imshow(f"{window_name} {cam_id}", dst)

def mouser(event, x, y, flags, param):
    undistort(x, y)

def main():
    for cam_id in cameras: get_images(cam_id)
    for cam_id in cameras:
        cv2.namedWindow(f"{window_name} {cam_id}")
        cv2.setMouseCallback(f"{window_name} {cam_id}", mouser)
    undistort(inputs["width"]/2, inputs["height"]/2)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
