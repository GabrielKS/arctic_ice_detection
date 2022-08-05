"""
Run the seg2info pipeline on a bunch of files and cache some results so we can use them quickly later.
Actually saving the transformed maps and such seems to be very storage-intensive, so we just save statistics.
"""

import os
import numpy as np
import cv2
from cv_experiments.seg2info import Seg2Info
s2i = Seg2Info()
import cv2
import pickle

# seginput_path = "../Sources/fuller_flat"
# segmap_path = "../inferred/one"
# output_dir = "../cached/one"

seginput_path = "../arctic_images_original_2/seginput"
segmap_path = "../arctic_images_original_2/segmaps"
output_dir = "../cached/first_ground"

# seginput_path = "../arctic_images_original_2/seginput"
# segmap_path = "../inferred/first_predicted"
# output_dir = "../cached/first_predicted"

def closest_ice(logpolar_plot, ice_thresh=0.1, nan_thresh=0.1, close_thresh=None):
    if close_thresh is None: close_thresh = s2i.cam_props["near_distance"]*4
    num = np.nansum(logpolar_plot, axis=1)
    n_nonnan = (~np.isnan(logpolar_plot)).sum(axis=1)
    frac_ice = np.where(n_nonnan/logpolar_plot.shape[1] > nan_thresh, np.divide(num, n_nonnan, where=(n_nonnan != 0)), np.nan)
    is_ice = frac_ice > ice_thresh
    i_last = -np.Inf if np.max(is_ice) == False else logpolar_plot.shape[0]-np.argmax(is_ice[::-1])-1
    dist_last = s2i.y2dist(i_last)
    return np.clip(dist_last, close_thresh, None)

def ice_amount(adjusted, scale, height, window_width_frac=0.5, window_height_frac=0.25):
    window_width_px = window_width_frac*adjusted.shape[1]*scale
    top = s2i.proc_props["sky_buffer"]*s2i.proc_props["upscale_factor"]
    bottom = int(top+window_height_frac*height*scale)
    left = int(adjusted.shape[1]/2-window_width_px/2)
    right = int(adjusted.shape[1]/2+window_width_px/2)
    subset = s2i.four_to_one(adjusted[top:bottom, left:right])

    n_nonnan = (~np.isnan(subset)).sum()  # Exclude invalid data from the denominator
    n_ice = np.nansum(subset)  # Each pixel is iciness from 0 to 1, so we can get total ice in pixels simply by summing
    return n_ice/n_nonnan

def process_one(image, outfile):
    to_save = {"name": image["name"]}
    img = image["segmap"]
    img = s2i.one_hot_four(img)
    img = s2i.upscale(img)
    img = s2i.undistort(img)
    # to_save["undistort"] = img
    horizon = s2i.find_horizon(img)
    to_save["horizon"] = horizon
    img, scale, height = s2i.rotate_image(img, horizon)
    to_save["scale"] = scale
    to_save["height"] = height
    img = s2i.adjust_and_crop(img, horizon, height)
    img = s2i.horizon_blur(img)
    # to_save["adjusted"] = img
    to_save["ice_amount"] = ice_amount(img, scale, height)
    img = s2i.camera_to_log_polar(img, scale, height)
    img = s2i.four_to_one(img)
    # to_save["logpolar"] = img
    to_save["closest_ice"] = closest_ice(img)
    pickle.dump(to_save, open(outfile, "wb"))

def load_dirs(seginput_path, segmap_path, segmap_ext = ".png", filter_fn = None, sort_fn = lambda name: name, max_n = None):
    """Like s2i.load_dirs but yields instead of returning a list and skips reading seginputs (but still looks for them)"""
    input_exts = (".png", ".jpg", ".jpeg")
    images = []
    for i, path in enumerate(sorted(os.listdir(seginput_path), key=sort_fn)):
        if not os.path.exists(os.path.join(seginput_path, path)): continue
        name = os.path.basename(path)
        this_segmap_path = os.path.join(segmap_path, os.path.splitext(name)[0]+segmap_ext)
        if not os.path.exists(this_segmap_path): continue  # Skip silently if either file doesn't exist
        if not path.endswith(input_exts): continue
        if filter_fn is not None and not filter_fn(path): continue
        yield {"name": name, "seginput": None, "segmap":
            cv2.imread(this_segmap_path, cv2.IMREAD_GRAYSCALE)}
        if max_n is not None and i >= max_n: break

def main():
    # images = load_dirs(seginput_path, segmap_path, segmap_ext=".jpg")
    images = load_dirs(seginput_path, segmap_path)
    for i, image in enumerate(images):
        process_one(image, os.path.join(output_dir, image["name"][:-4]+".pkl"))
        if i % 10 == 0: print(f"{i}")

if __name__ == "__main__":
    main()
