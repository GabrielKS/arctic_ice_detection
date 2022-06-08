import get_data

import tensorflow as tf
keras = tf.keras

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout

n_epochs = 50
batch_size = 16

def transfer_model(base, addon):
    base.summary()
    addon.summary()
    base.trainable = False
    model = Sequential()
    model.add(base)
    model.add(addon)
    # inn = keras.Input(shape=input_shape)
    # model = keras.Model(inputs = inn, outputs = addon(base(inn)))
    model.summary()
    return model

def original_addon(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    return model

def vgg16_base(input_shape):
    return VGG16(include_top=False, weights="imagenet", input_shape=input_shape)

def inceptionv3_base(input_shape):
    return InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape)

def vgg16_transfer_model(input_shape):
    base = vgg16_base(input_shape)
    addon = original_addon(base.output_shape[1:])
    return transfer_model(base, addon)

def inceptionv3_transfer_model(input_shape):
    base = inceptionv3_base(input_shape)
    addon = original_addon(base.output_shape[1:])
    return transfer_model(base, addon)

def train(model, epochs, base_fn=None):
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    train_ds = get_processed_dataset(get_data.train_label, to_augment=True)
    valid_ds = get_processed_dataset(get_data.valid_label)
    if base_fn is not None:
        train_ds = base_fn(train_ds)
        valid_ds = base_fn(valid_ds)
    model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=valid_ds)

def test(model, base_fn=None):
    test_ds = get_processed_dataset(get_data.test_label)
    if base_fn is not None:
        test_ds = base_fn(test_ds)
    eval_results = model.evaluate(test_ds, verbose=1)
    predict_results = model.predict(test_ds, verbose=1)
    return eval_results, predict_results

def apply_pipeline(pipeline, ds):
    return ds.map(lambda x, y: (pipeline(x), y))

def preprocess(ds):
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

def get_processed_dataset(label, to_augment=False):
    ds = get_data.get_raw_dataset(label, batch_size=batch_size)
    ds = preprocess(ds)
    if to_augment: ds = augment(ds)
    return ds

# Evaluate the lazy pipeline now such that
#   (a) we only run the full dataset through the base model once, for performance reasons, and
#   (b) the output of the base model and the labels all correspond to the same random choices
#       in the augmentation phase.
# It seems like this should already exist.
def eager_evaluate_xy_pipeline(ds):
    x_batches, y_batches = [], []
    batch_size = None
    n_batches = ds.cardinality().numpy()
    print(f"Eagerly evaluating input pipeline in {n_batches} batches:")
    for i, (x_batch, y_batch) in enumerate(ds):
        x_batches.append(x_batch)
        y_batches.append(y_batch)
        if batch_size is None: batch_size = x_batch.shape[0]
        print(i, end=("\n" if i == n_batches-1 else ".. "), flush=True)
    return tf.data.Dataset.from_tensor_slices(
        (tf.concat(x_batches, axis=0), tf.concat(y_batches, axis=0))).batch(batch_size)

def main():
    base_model = vgg16_base((*get_data.image_size, 3))
    # We can run the base model once ahead of time for a significant speedup
    base_fn = lambda ds: eager_evaluate_xy_pipeline(apply_pipeline(base_model, ds))
    model = original_addon(base_model.output_shape[1:])
    train(model, n_epochs, base_fn=base_fn)
    print(test(model, base_fn=base_fn))

    # model = vgg16_transfer_model((*get_data.image_size, 3))
    # train(model, n_epochs)
    # print(test(model))

if __name__ == "__main__":
    main()
