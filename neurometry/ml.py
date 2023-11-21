import pandas as pd
import numpy as np
import tifffile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
#mport tensorflow as tf
from albumentations import (
    Compose, Rotate, VerticalFlip, HorizontalFlip, ElasticTransform, Transpose, Sharpen
)
import random
random.seed(123)

transforms = Compose([
            Rotate(),
            VerticalFlip(),
            Transpose(),
            HorizontalFlip(),
            #Sharpen()
            #ElasticTransform()
        ])

def ingestor(df):
    id_to_vol = {}
    for index, row in df.iterrows():
        name = row["wk_id"]
        vol1 = np.transpose(tifffile.imread("cutouts/"+name+"/img/vol.tiff"), (1,2,0))
        seg1 = np.transpose(tifffile.imread("cutouts/"+name+"/presyn/presyn.tiff"), (1,2,0))
        masked1 = (seg1/255)*vol1
        #id_to_vol[name] = masked1
        #Divide by 255 here
        id_to_vol[name] = np.array(masked1/255.0).astype('float32')
    
    return id_to_vol

def change_target(row):
    #completely random
    #if random.uniform(0, 1) < .25:
    #    return 0
    #else:
    #    return 1
    #switching target variables, NONE should always be 0, anything else 1
    if row["pre_syn_label"] == "NONE" or row["pre_syn_label"] == "FEW":
        return 0
    else:
        return 1
    #elif row["pre_syn_label"] == "FEW":
    #    return 1
    #else:
    #    return 2

def aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    #aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_img = tf.reshape(aug_img, [160,160,16])
    aug_img = tf.expand_dims(aug_img, axis=3)
    return aug_img

def process_data(volume, label):
    aug_volume = tf.numpy_function(func=aug_fn, inp=[volume], Tout=tf.float32)
    return aug_volume, label

def train_preprocessing(volume, label):
    """Process training data."""
    # Divide by 255
    #volume = volume/255
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data."""
    #volume = volume/255
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_model(width=160, height=160, depth=16):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))



    x = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same")(inputs)
    #x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same")(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Flatten()(x)
    #x = layers.GlobalMaxPooling3D()(x)
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    #outputs = layers.Dense(units=3, activation="softmax")(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn-try4")
    return model

def loader(filepath):
    model = load_model(filepath)
    return model