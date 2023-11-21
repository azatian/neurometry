# %%
import pandas as pd
import os
from datetime import datetime
from neurometry import viz, ml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# %%
print("-----------STARTING PIPELINE---------------------------")
df = pd.read_csv("data/dcvsyn600_annotated_12112022.csv")

# %%
print("-----------DATAFRAME LOAD COMPLETE---------------------")
df["target"] = df.apply(ml.change_target, axis=1)
#print(df["target"].value_counts())
print("-----------DATAFRAME TARGET ALTERATION COMPLETE--------")
id_to_vol = ml.ingestor(df)

# %%
sample_img = id_to_vol['dcvsyn1']
sample_img_two = id_to_vol['dcvsyn4']

# %%
from neurometry import ic

# %%
from albumentations import (
    Compose, Rotate, VerticalFlip, HorizontalFlip, ElasticTransform, Transpose
)

# %%
