import os

import get_data
import preprocess
from preprocess import apply_pipeline, get_preprocessed_dataset
import base_models
from base_models import load_base_modeled

import tensorflow as tf
keras = tf.keras
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout

n_epochs = 50
batch_size = preprocess.batch_size

def transfer_model(base, addon):
    base.trainable = False
    model = Sequential()
    model.add(base)
    model.add(addon)
    return model

def original_addon(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    return model

def vgg16_transfer_model(input_shape):
    base = base_models.vgg16_base(input_shape)
    addon = original_addon(base.output_shape[1:])
    return transfer_model(base, addon)

def inceptionv3_transfer_model(input_shape):
    base = base_models.inceptionv3_base(input_shape)
    addon = original_addon(base.output_shape[1:])
    return transfer_model(base, addon)

def train(model, epochs, batch_size, train_ds, valid_ds, base_fn=None, steps_per_epoch=None):
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    if base_fn is not None:
        train_ds = base_fn(train_ds)
        valid_ds = base_fn(valid_ds)
    return model.fit(train_ds, validation_data=valid_ds, epochs=epochs,
        batch_size=batch_size, steps_per_epoch=steps_per_epoch)

def test(model, test_ds, base_fn=None):
    if base_fn is not None:
        test_ds = base_fn(test_ds)
    eval_results = model.evaluate(test_ds, verbose=1)
    predict_results = model.predict(test_ds, verbose=1)
    predict_results, ground_truth, file_paths = eager_evaluate_xy_pipeline(
        apply_pipeline(model, test_ds), make_dataset=False)
    return {
        "history": dict(zip(model.metrics_names, eval_results)),
        "prediction": predict_results,
        "actual": ground_truth,
        "filenames": file_paths}

def print_test_results(results, threshold=0.5):
    for k in results["history"]: print(f"{k}: {results['history'][k]}")
    print(results["prediction"].shape)
    for i, pred in enumerate(results["prediction"]):
        fpath = results["filenames"][i]
        nicename = os.path.join(os.path.basename(os.path.dirname(fpath)), os.path.basename(fpath))
        pred = results["prediction"][i][0]
        actual = results["actual"][i][0]
        correct = (pred < threshold) if (actual < threshold) else (pred >= threshold)
        print(f"{nicename}\t{pred:.2f}\t{'WRONG' if not correct else ''}")

# Evaluate the lazy pipeline of (inputs, ground truths) now! For input pipelines, we might do this so that:
#   (a) we only run the full dataset through the base model once, for performance reasons, and
#   (b) the output of the base model and the labels all correspond to the same random choices
#       in the augmentation phase.
# It seems like this should already exist.
def eager_evaluate_xy_pipeline(ds, make_dataset=True):
    file_paths = ds.file_paths
    x_batches, y_batches = [], []
    batch_size = None
    n_batches = ds.cardinality().numpy()

    print(f"Eagerly evaluating input pipeline in {n_batches} batches:")
    # It seems we must resort to a loop here. Hopefully the data are batched such that this isn't too bad.
    # Parallelization options are set up where the pipeline is created.
    for i, (x_batch, y_batch) in enumerate(ds):
        x_batches.append(x_batch)
        y_batches.append(y_batch)
        if batch_size is None: batch_size = x_batch.shape[0]
        print(i, end=("\n" if i == n_batches-1 else ".. "), flush=True)
    x_tensor, y_tensor = tf.concat(x_batches, axis=0), tf.concat(y_batches, axis=0)
    if not make_dataset: return x_tensor, y_tensor, file_paths
    ds = tf.data.Dataset.from_tensor_slices(
        (x_tensor, y_tensor)).batch(batch_size)
    ds.file_paths = file_paths
    return ds

# A demo of how our new module layout allows us to easily do transfer learning in a variety of configurations
def main():
    print("Applying the original addon to cached augmentation -> VGG16 output")
    train_ds_1 = load_base_modeled(base_models.vgg16_label, get_data.train_label)
    valid_ds_1 = load_base_modeled(base_models.vgg16_label, get_data.valid_label)
    test_ds_1 = load_base_modeled(base_models.vgg16_label, get_data.test_label)
    model_1 = original_addon(train_ds_1.element_spec[0].shape[1:])
    # Approximate only running one augmentation repetition through each epoch (see preprocess.augmentation_reps)
    spe_1 = tf.data.experimental.cardinality(train_ds_1).numpy()/preprocess.augmentation_reps
    train(model_1, n_epochs, batch_size, train_ds_1.repeat(), valid_ds_1, steps_per_epoch=spe_1)
    results_1 = test(model_1, test_ds_1)
    print_test_results(results_1)

    print("Running VGG16 once at the beginning on the fly")
    base = base_models.vgg16_base((*get_data.image_size, 3))
    otf_pipeline = lambda ds: eager_evaluate_xy_pipeline(apply_pipeline(base, ds))
    # Note that in this case (which corresponds to the original), the train dataset is only passed
    # through augmentation once, which at least partially defeats the purpose.
    train_ds_2 = otf_pipeline(get_preprocessed_dataset(get_data.train_label, to_augment=True, shuffle=True))
    valid_ds_2 = otf_pipeline(get_preprocessed_dataset(get_data.valid_label, shuffle=True))
    test_ds_2 = otf_pipeline(get_preprocessed_dataset(get_data.test_label, shuffle=False))
    model_2 = original_addon(train_ds_2.element_spec[0].shape[1:])
    train(model_2, n_epochs, batch_size, train_ds_2, valid_ds_2)
    results_2 = test(model_2, test_ds_2)
    print_test_results(results_2)

    print("Running everything at all times")
    model_3 = vgg16_transfer_model((*get_data.image_size, 3))
    train_ds_3 = get_preprocessed_dataset(get_data.train_label, to_augment=True, shuffle=True)
    valid_ds_3 = get_preprocessed_dataset(get_data.valid_label, shuffle=True)
    test_ds_3 = get_preprocessed_dataset(get_data.test_label, shuffle=False)
    train(model_3, n_epochs, batch_size, train_ds_3, valid_ds_3)
    results_3 = test(model_3, test_ds_3)
    print_test_results(results_3)

if __name__ == "__main__":
    main()
