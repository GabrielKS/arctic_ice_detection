import get_data

import tensorflow as tf
keras = tf.keras

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout

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

def original_augment(input_ds):
    return input_ds  # TODO: implement data augmentation

def train(model, epochs):
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    train_ds = get_data.get_dataset(get_data.train_label)
    valid_ds = get_data.get_dataset(get_data.valid_label)
    model.fit(train_ds, epochs=epochs, validation_data=valid_ds)

def test(model):
    test_ds = get_data.get_dataset(get_data.test_label)
    eval_results = model.evaluate(test_ds, verbose=1)
    predict_results = model.predict(test_ds, verbose=1)

def main():
    model = vgg16_transfer_model((*get_data.image_size, 3))
    train(model, 50)
    print(test(model))

if __name__ == "__main__":
    main()
