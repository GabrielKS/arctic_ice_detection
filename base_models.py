import os

import get_data
from get_data import reset_dir
from preprocess import apply_pipeline, load_preprocessed, save_file_paths, load_file_paths

import tensorflow as tf
keras = tf.keras

base_modeled_root = os.path.abspath("../models/base_modeled")
vgg16_label, inceptionv3_label, xception_label = "vgg16", "inceptionv3", "xception"
input_shape = (224, 224, 3)

# Returns a predictor function that preprocesses data, configures a built-in model, and runs it
def builtin_base_with_preprocess(input_shape, native_shape, preprocess, model):
    def f(data):
        if data.shape[1:] == input_shape: print("No resizing necessary")
        else:
            data = keras.layers.Resizing(*input_shape[:-1])(data)
        if data.shape[1:] == native_shape:
            print("Native size!")
            shape_arg = None
        else: shape_arg = input_shape
        data = preprocess(data)
        return model(include_top=False, weights="imagenet", input_shape=shape_arg)(data)
    return f

vgg16_base = lambda input_shape: builtin_base_with_preprocess(input_shape, (224, 224, 3),
    tf.keras.applications.vgg16.preprocess_input, keras.applications.vgg16.VGG16)

inceptionv3_base = lambda input_shape: builtin_base_with_preprocess(input_shape, (299, 299, 3),
    tf.keras.applications.inception_v3.preprocess_input, keras.applications.inception_v3.InceptionV3)

xception_base = lambda input_shape: builtin_base_with_preprocess(input_shape, (299, 299, 3),
    tf.keras.applications.xception.preprocess_input, keras.applications.xception.Xception)

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

base_models_to_run = [(vgg16_label, vgg16_base)]

def main():
    print("Deleting previous results…")
    reset_dir(base_modeled_root)

    # The unconventional ordering is so we do fastest first
    # so we have some idea of how long the longer operations will take
    for label, model_fn in base_models_to_run:
        print("Prerunning model "+label)
        model = model_fn(input_shape)
        print(f"Test data…")
        base_model(label, get_data.test_label, model)
        print("Validation data…")
        base_model(label, get_data.valid_label, model)
        print("Training data…")
        base_model(label, get_data.train_label, model)
        print("Done with "+label)

if __name__ == "__main__":
    main()
