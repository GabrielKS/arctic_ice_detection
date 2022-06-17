import os
import time

import get_data
from get_data import reset_dir
from evaluate_results import format_time_estimate, format_finished_msg

import tensorflow as tf
keras = tf.keras
from keras import Sequential

augmentation_reps = 10
batch_size = 16

preprocessed_root = os.path.abspath("../models/preprocessed")

compression = None  # GZIP compression saves significant space but takes significant time

# Apply a Sequential of pipeline steps to a Dataset that originated with a set of files, preserving file_paths
def apply_pipeline(pipeline, ds, preserve_file_paths = True):
    if preserve_file_paths: file_paths = ds.file_paths
    # Let Tensorflow parallelize for us.
    # This leads to some weirdness with the status messages but produces a significant speedup.
    ds = ds.map(lambda x, y: (pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if preserve_file_paths: ds.file_paths = file_paths
    return ds

def scale(ds):
    return apply_pipeline(Sequential([
        keras.layers.Rescaling(1./255)
        # can insert more here
    ]), ds)

def augment(ds):
    return apply_pipeline(Sequential([
        keras.layers.RandomFlip("horizontal", seed=get_data.randomseed),
        keras.layers.RandomRotation(0.2, seed=get_data.randomseed)
        # can insert more here
    ]), ds)

def get_preprocessed_dataset(label, to_augment=False, shuffle=False):
    ds = get_data.get_raw_dataset(label, batch_size=batch_size, shuffle=shuffle)
    file_paths = ds.file_paths
    # ds = scale(ds)  # Nah, let's do this later on a per-model basis
    if to_augment: ds = augment(ds)
    ds.file_paths = file_paths
    return ds

def save_file_paths(root, label, file_paths):
    with open(os.path.join(root, label+"_file_paths.txt"), "w") as f:
        f.write("\n".join(file_paths)+"\n")

def load_file_paths(root, label):
    with open(os.path.join(root, label+"_file_paths.txt"), "r") as f:
        file_paths = list(filter(lambda s: len(s) > 0,  # Strip empty lines
            map(lambda s: s.rstrip(), f.readlines())))  # Strip trailing newlines
    return file_paths

def preprocess_dataset(label, shuffle, to_augment, reps=1, t_per_batch=None):
    ds = get_preprocessed_dataset(label, to_augment, shuffle)
    file_paths = ds.file_paths
    ds = ds.repeat(reps)
    ds.file_paths = file_paths
    n_batches = tf.data.experimental.cardinality(ds).numpy()

    status = f"Preprocessing on {n_batches} batches. "
    status += format_time_estimate(None if t_per_batch is None else t_per_batch*n_batches)
    print(status)
    t0 = time.time()

    save_preprocessed(ds, label)
    
    t1 = time.time()
    print(format_finished_msg(t1-t0))
    return (t1-t0)/n_batches

def save_preprocessed(ds, label):
    save_file_paths(preprocessed_root, label, ds.file_paths)
    tf.data.experimental.save(ds, os.path.join(preprocessed_root, label), compression=compression)

def load_preprocessed(label):
    ds = tf.data.experimental.load(os.path.join(preprocessed_root, label), compression=compression)
    ds.file_paths = load_file_paths(preprocessed_root, label)
    return ds

def main():
    print("Deleting previous results…")
    reset_dir(preprocessed_root)
    print("Preprocessing test data…")
    t_per_batch = preprocess_dataset(get_data.test_label,  shuffle=False, to_augment=False)
    print("Preprocessing validation data…")
    t_per_batch += preprocess_dataset(get_data.valid_label, shuffle=True, to_augment=False, t_per_batch=t_per_batch)
    t_per_batch /= 2
    print("Preprocessing training data…")
    preprocess_dataset(get_data.train_label, shuffle=True, to_augment=True, reps=augmentation_reps, t_per_batch=t_per_batch)
    print("Done preprocessing.")

if __name__ == "__main__":
    main()
