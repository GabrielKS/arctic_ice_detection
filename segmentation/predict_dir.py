"""Load a previously trained model and run inference on a list of directories of input."""

import os
from fastai.vision.all import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

# input_dirs = [f.path for f in os.scandir("../Sources/Raw/touse/") if f.is_dir()]
# input_dirs = [os.path.abspath("../arctic_images_original_2/seginput")]

# input_dirs = ["../Sources/fuller_flat"]
# output_dir = "../inferred/one"
# model_path = "../saved_models/export_2022-08-01T20-46-50.pkl"

input_dirs = [os.path.abspath("../arctic_images_original_2/seginput")]
output_dir = "../inferred/first_predictedd"
model_path = "../saved_models/export_2022-08-01T20-46-50.pkl"

def seginput2segmap(): raise NotImplementedError  # load_learner thinks it needs this

def remove_whitespace():  # sheesh
  plt.gcf().add_axes(plt.Axes(plt.gcf(), [0,0,1,1]))
  plt.axis("off")

def plt_superimposed(base, mask):
  plt.imshow(base)
  plt.imshow(mask, alpha=0.25, cmap="tab20", vmax=4)

def process_one(learn, i, img, outfile):
    img = Resize((279, 500), method=ResizeMethod.Squish)(img)
    pred = learn.predict(img)[0].numpy()
    cv2.imwrite(outfile, pred)
    return

def main():
   # TODO certainly there is a less ugly way to do all of this with fastai
    # with open("../saved_models/validlist_2022-08-01T23-30-48.txt") as validfile:
    #     validset = set(validfile.read().splitlines())
    learn = load_learner(model_path)
    images = sorted(f.path for input_dir in input_dirs
        for f in os.scandir(input_dir) if f.is_file()
            and os.path.basename(f.path)[0] != '.'
            and os.path.basename(f.path)[-4:] == ".jpg")
            # and (f.name in validset)
    dummy_labels = np.arange(len(images))
    dls = ImageDataLoaders.from_lists("", images, dummy_labels, valid_pct=0)
    for i, ((img, _), (fname, _)) in enumerate(zip(dls.train_ds, dls.train.items)):
        process_one(learn, i, img, os.path.join(output_dir, os.path.basename(fname)))
        if i % 10 == 0: print(f"{i}/{len(dls.train.items)}")

if __name__ == "__main__":
    main()
