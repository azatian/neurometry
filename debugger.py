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
df = pd.read_csv("data/dcvsyn600_annotated_12112022.csv")

# %%
df["target"] = df.apply(ml.change_target, axis=1)

# %%
id_to_vol = ml.ingestor(df)

# %%
X_train, X_val, Y_train, Y_val = train_test_split(df["wk_id"], df["target"], test_size=.2, random_state=49, stratify=df["target"])
X_train = np.array(X_train)
X_val = np.array(X_val)
Y_train = np.array(Y_train)
Y_val = np.array(Y_val)

# %%
Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))

# %%
X_train_data = np.array([id_to_vol[x] for x in X_train])
X_val_data = np.array([id_to_vol[x] for x in X_val])

# %%
print("-----------TRAIN/VALIDATION SPLIT COMPLETE-------------")
train_loader = tf.data.Dataset.from_tensor_slices((X_train_data, Y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((X_val_data, Y_val))
print("-----------TENSORFLOW DATA LOADING COMPLETE------------")

# %%
batch_size = 2
    #print(len(X_train_data))
    #print(len(X_val_data))
    
train_dataset = (
    train_loader.shuffle(len(X_train_data))
    .map(ml.process_data)
    .batch(batch_size)
    .prefetch(2)
)

validation_dataset = (
    validation_loader.shuffle(len(X_val_data))
    .map(ml.validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

# %%
vols, labels = list(train_dataset.take(1))[0]
plt.figure()
plt.imshow(vols[1].numpy()[:,:,2,0], cmap=plt.cm.gray)
plt.show()