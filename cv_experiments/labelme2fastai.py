"""
A small utility to derive segmentation PNGs that fastai will like from the JSONs labelme gives us.
Can also perform undistortion and basic validation.
Quick and dirty, not at all optimized.
"""

from labelme.utils import img_b64_to_arr, shapes_to_label
import os
import json
import cv2
import numpy as np
import copy
import cv_experiments.cv_common as cc

background_label = "water"  # What to call things that aren't part of a shape
# label2value = {"water": 0, "sky": 100, "ice": 200, "other": 255}  # Good for humans
# label2value = {"water": 0, "sky": 255, "ice": 63, "other": 127}  # Good for undistortion tuning
label2value = {"water": 0, "sky": 1, "ice": 2, "other": 3}  # Good for fastai
round_thresh = 5  # If a value is within this many pixels of an edge of the image, snap it to that edge

input_dirs = [os.path.abspath("../arctic_images_original_2/no_ice"), os.path.abspath("../arctic_images_original_2/ice")]
segmap_output = os.path.abspath("../arctic_images_original_2/segmaps")
seginput_output = os.path.abspath("../arctic_images_original_2/seginput")

# It is hard within the labelme interface to get a point right up against the edge, and we often want to do this
def snap_points(shapes, imgshape):
    height, width = imgshape[:2]
    shapes = copy.deepcopy(shapes)
    for shape in shapes:
        for point in shape["points"]:
            if point[0] < round_thresh: point[0] = 0
            if point[0] > width-round_thresh: point[0] = width
            if point[1] < round_thresh: point[1] = 0
            if point[1] > height-round_thresh: point[1] = height
    return shapes

# Shut things down or just warn us if a condition is not met
def warn_if_not(condition, msg, fname):
    fname = os.path.basename(fname)
    # assert condition, f"{fname}: {msg}"
    if not condition: print(f"WARNING: {fname}: {msg}")

# Some sanity checks to make sure things look as they should
def validate(shapes, classes, fname):
    n_skies = len(list(filter(lambda shape: shape["label"] == "sky", shapes)))
    warn_if_not(n_skies == 1, f"has {n_skies} sky shapes", fname)  # Each image should have exactly 1 sky shape
    warn_if_not(classes[0, 0] == label2value["sky"], "top left is not sky", fname)
    warn_if_not(classes[0, -1] == label2value["sky"], "top right is not sky", fname)
    warn_if_not(classes[-1, 0] != label2value["sky"], "bottom left is sky", fname)
    warn_if_not(classes[-1, -1] != label2value["sky"], "bottom right is sky", fname)

def process_one(path_in, path_segmap_out, path_seginput_out):
    data = json.load(open(path_in))
    img = img_b64_to_arr(data.get("imageData"))
    shapes = snap_points(data["shapes"], img.shape)  # The meat of it is in here
    classes, _ = shapes_to_label(img.shape, shapes, label2value)
    classes = np.where(classes == 0, label2value[background_label], classes).astype(np.uint8)
    validate(shapes, classes, path_in)
    classes = cc.undistort(classes, "new")
    cv2.imwrite(path_segmap_out, classes)
    cv2.imwrite(path_seginput_out, cc.undistort(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), "new"))

def main():
    jsons = sorted(f.path for input_dir in input_dirs
        for f in os.scandir(input_dir) if f.is_file()
            and os.path.basename(f.path)[0] != '.'
            and os.path.basename(f.path)[-5:] == ".json")
    for fname in jsons:
        process_one(fname, os.path.join(segmap_output, os.path.basename(fname[:-5]+".png")),
            os.path.join(seginput_output, os.path.basename(fname[:-5]+".jpg")))

if __name__ == "__main__":
    main()
