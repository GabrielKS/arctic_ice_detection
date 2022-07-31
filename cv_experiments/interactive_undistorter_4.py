# The usual interactive undistorter plus scaling

import os
import numpy as np
import cv2

input_dir = os.path.abspath("../representatives/large_distortion")
window_name = "Interactive Undistorter"

# Settings that may need to be adjusted depending on the input:
crop = slice(None)  # No cropping
# crop = [slice(100, -100), slice(200, -200)]  # Some cropping
xrange, yrange = 7, 6  # How many powers of ten to move across in each direction
xoffset, yoffset = -10, -10  # Minimum power of ten we care about
threshold_image = False  # Whether to apply some thresholding to the image
focal_length = 10*3.84

inputs = {"sources": None, "combo": None, "width": None, "height": None}

def get_images():
    # images = s2i.load_dirs("../representatives/segmentation/seginput/other", "../arctic_images_original_2/segmaps")
    # inputs["sources"] = [image["segmap"] for image in images]
    inputs["sources"] = [cv2.imread(os.path.join(input_dir, input_img), cv2.IMREAD_GRAYSCALE)
        for input_img in os.listdir(input_dir) if input_img[-4:] == ".jpg"]
    for i in range(len(inputs["sources"])):
        inputs["sources"][i] = inputs["sources"][i][crop]
    
    # Assumes all images are the same size
    inputs["width"] = inputs["sources"][0].shape[1]
    inputs["height"] = inputs["sources"][0].shape[0]
    print(inputs["width"])
    print(inputs["height"])

    inputs["combo"] = np.zeros_like(inputs["sources"][0])
    n_inputs = len(inputs["sources"]) 
    for this_src in inputs["sources"]:
        if threshold_image:
            this_src = cv2.GaussianBlur(this_src, (15, 15), 0)
            # _,this_src = cv2.threshold(this_src, 140, 255, cv2.THRESH_BINARY)
            this_src = cv2.adaptiveThreshold(this_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, inputs["height"]//2*2+1, 0)
        inputs["combo"] += (this_src/n_inputs).astype("uint8")

def undistort(x, y):
    # Exponential with sign added: greater towards edges, close to zero at center, down+right is positive
    coeff_1  = np.sign(x-inputs["width"]/2)*10**(np.abs((x-inputs["width"]/2)/(inputs["width"]/2)*xrange)+xoffset)
    coeff_2 = np.sign(y-inputs["height"]/2)*10**(np.abs((y-inputs["height"]/2)/(inputs["height"]/2)*yrange)+yoffset)
    coeff_1, coeff_2 = -5.35e-4, 3.05e-7
    print(f"{coeff_1:.2E}, {coeff_2:.2E}")  # Write these down when you find a good undistortion

    # Experimental evidence suggests:
    # Parameter 1: radially symmetric, affects the outer edges more
    # Parameter 2: radially symmetric, affects the outer edges way more
    # Parameter 3: pitch
    # Parameter 4: yaw
    distCoeffs = np.array([coeff_1, coeff_2, 0, 0])
    cam = np.array([
        [focal_length,  0,              inputs["width"]/2],
        [0,             focal_length,   inputs["height"]/2],
        [0,             0,              1]])
    
    height, width = inputs["combo"].shape[:2]
    sf = 0.7845    # -5.35e-4, 3.05e-7
    map1, map2 = cv2.initUndistortRectifyMap(cam, distCoeffs, np.eye(3), cam,
        (inputs["combo"].shape[1], inputs["combo"].shape[0]), cv2.CV_32FC1)
    map1 = map1/sf-width*(1/sf-1)/2
    map2 = map2/sf-height*(1/sf-1)/2
    dst = cv2.remap(inputs["combo"], map1, map2, interpolation=cv2.INTER_NEAREST)
    # dst = cv2.undistort(inputs["combo"], cam, distCoeffs)
    cv2.imshow(window_name, dst)

def mouser(event, x, y, flags, param):
    undistort(x, y)

def main():
    get_images()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouser)
    undistort(inputs["width"]/2, inputs["height"]/2)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
