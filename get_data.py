import os
import shutil
import sklearn.model_selection

input_root = os.path.abspath("../arctic_images")  # Where to find the raw input
ice_label, noice_label = "ice", "no_ice"
input_exts = (".png", ".jpg", ".jpeg")  # Just a basic filter to exclude some non-image files

output_root = os.path.abspath("../models/input")  # Where to put the structured output
train_label, valid_label, test_label = "train", "validate", "test"

train_frac = 0.8  # Of total input, reserve 1-this for final testing
valid_frac = 0.25 # Of train, reserve this for validation

def get_images(condition):
    input_path = os.path.join(input_root, condition)
    files = [os.path.join(input_path, fname) for fname in os.listdir(input_path)]
    files = filter(os.path.isfile, files)
    files = filter(lambda f: f.lower().endswith(input_exts), files)
    return list(files)

# The heart of the script: use sklearn's train_test_split to randomly segment the files into
# training, validation, and testing datasets
def train_valid_test_split(files):
    train_valid, test = sklearn.model_selection.train_test_split(files, train_size=train_frac)
    train, valid = sklearn.model_selection.train_test_split(train_valid, test_size=valid_frac)
    return train, valid, test

def save_images(path, images):
    if not os.path.exists(path): os.makedirs(path)
    # Remove existing so files don't pile up if re-run
    for file in os.listdir(path): os.remove(os.path.join(path, file))
    for img in images: shutil.copy2(img, path)

def save_condition(condition, train, valid, test):
    for pair in ((train_label, train), (valid_label, valid), (test_label, test)):
        save_images(os.path.join(output_root, pair[0], condition), pair[1])
        print(f"{condition}: {pair[0]}: {len(pair[1])} images")
    print()

def sort_images():
    assert os.path.exists(output_root), "Could not find output dir at "+output_root
    save_condition(ice_label, *train_valid_test_split(get_images(ice_label)))
    save_condition(noice_label, *train_valid_test_split(get_images(noice_label)))

def main():
    sort_images()

if __name__ == "__main__":
    main()