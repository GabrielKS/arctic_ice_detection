import os

import get_data
from get_data import reset_dir
from preprocess import apply_pipeline, load_preprocessed, save_file_paths, load_file_paths

import tensorflow as tf
keras = tf.keras
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

base_modeled_root = os.path.abspath("../models/base_modeled")
vgg16_label, inceptionv3_label = "vgg16", "inceptionv3"

def vgg16_base(input_shape):
    return VGG16(include_top=False, weights="imagenet", input_shape=input_shape)

def inceptionv3_base(input_shape):
    return InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape)

def base_model(model_label, split_label, model):
    ds = load_preprocessed(split_label)
    print(f"Predicting on {tf.data.experimental.cardinality(ds)} batches")
    ds = apply_pipeline(model, ds)
    save_base_modeled(ds, model_label, split_label)

def save_base_modeled(ds, model_label, split_label):
    tf.data.experimental.save(ds, os.path.join(base_modeled_root, model_label, split_label))
    save_file_paths(base_modeled_root, split_label, ds.file_paths)

def load_base_modeled(model_label, split_label):
    ds = tf.data.experimental.load(os.path.join(base_modeled_root, model_label, split_label))
    ds.file_paths = load_file_paths(base_modeled_root, split_label)
    return ds

def main():
    print("Deleting previous results…")
    reset_dir(base_modeled_root)

    # The unconventional ordering is so we do fastest first
    # so we have some idea of how long the longer operations will take
    for label, model_fn in [(inceptionv3_label, inceptionv3_base), (vgg16_label, vgg16_base)]:
        print("Prerunning model "+label)
        model = model_fn((*get_data.image_size, 3))
        print(f"Test data…")
        base_model(label, get_data.test_label, model)
        print("Validation data…")
        base_model(label, get_data.valid_label, model)
        print("Training data…")
        base_model(label, get_data.train_label, model)
        print("Done with "+label)

if __name__ == "__main__":
    main()
