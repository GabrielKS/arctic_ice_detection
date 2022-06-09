import os
import shutil

from sklearn.model_selection import train_test_split
import tensorflow as tf
keras = tf.keras

randomseed = 0

input_root = os.path.abspath("../arctic_images")  # Where to find the raw input
image_size = (150, 150)  # TODO perhaps get this programmatically
conditions = ("no_ice", "ice")  # Index corresponds to desired binary model output (0 or 1)
input_exts = (".png", ".jpg", ".jpeg")  # Just a basic filter to exclude some non-image files

output_root = os.path.abspath("../models/input")  # Where to put the structured output
train_label, valid_label, test_label = "train", "validate", "test"  # TODO perhaps make an enum of this

train_frac = 0.8  # Of total input, reserve 1-this for final testing
valid_frac = 0.25 # Of train, reserve this for validation

def get_raw(condition):
    input_path = os.path.join(input_root, condition)
    files = [os.path.join(input_path, fname) for fname in os.listdir(input_path)]
    files = filter(os.path.isfile, files)
    files = filter(lambda f: f.lower().endswith(input_exts), files)
    shape = None
    return list(files)

# The heart of the script: use sklearn's train_test_split to randomly segment the files into
# training, validation, and testing datasets
def train_valid_test_split(files):
    train_valid, test = train_test_split(files, train_size=train_frac, random_state=randomseed)
    train, valid = train_test_split(train_valid, test_size=valid_frac, random_state=randomseed)
    return train, valid, test

def save_images(path, images):
    if not os.path.exists(path): os.makedirs(path)
    # Remove existing so files don't pile up if re-run
    for file in os.listdir(path): os.remove(os.path.join(path, file))
    for img in images: shutil.copy2(img, path)

def save_condition(condition, train, valid, test):
    for ds_label,ds in ((train_label, train), (valid_label, valid), (test_label, test)):
        save_images(os.path.join(output_root, ds_label, condition), ds)
        print(f"{condition}: {ds_label}: {len(ds)} images")
    print()

def reset_dir(dir, assert_exists=True):
    if assert_exists: assert os.path.exists(dir), "Could not find output dir at "+dir
    shutil.rmtree(dir, ignore_errors=True)
    if os.path.exists(dir): shutil.rmtree(dir)  # Try again
    os.mkdir(dir)
    # for file in os.listdir(dir):
    #     fpath = os.path.join(dir, file)
    #     if os.path.isfile(fpath): os.remove(fpath)
    #     else: shutil.rmtree(fpath)

def sort_images():
    reset_dir(output_root)
    for condition in conditions:
        save_condition(condition, *train_valid_test_split(get_raw(condition)))

def get_raw_dataset(ds_label, batch_size=32, shuffle=False):
    dir = os.path.join(output_root, ds_label)
    ds = keras.preprocessing.image_dataset_from_directory(dir,
        labels="inferred",
        label_mode="binary",
        class_names=tuple(conditions),  # To get the order right
        color_mode="rgb",
        image_size=image_size,
        shuffle=shuffle,
        seed=randomseed,
        batch_size=batch_size)
    return ds

def main():
    sort_images()

if __name__ == "__main__":
    main()
