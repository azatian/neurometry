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

def main() -> None:
    #The 600 synapses
    print("-----------STARTING PIPELINE---------------------------")
    df = pd.read_csv("data/dcvsyn600_annotated_12112022.csv")
    print("-----------DATAFRAME LOAD COMPLETE---------------------")
    df["target"] = df.apply(ml.change_target, axis=1)
    #print(df["target"].value_counts())
    print("-----------DATAFRAME TARGET ALTERATION COMPLETE--------")
    id_to_vol = ml.ingestor(df)
    print("-----------600 SYNAPSES SUCCESSFULLY LOADED------------")
    X_train, X_val, Y_train, Y_val = train_test_split(df["wk_id"], df["target"], test_size=.2, random_state=49, stratify=df["target"])
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    Y_train = np.array(Y_train)
    Y_val = np.array(Y_val)


    Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
    Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))
    #Y_train = to_categorical(Y_train)
    #Y_val = to_categorical(Y_val)
    
    X_train_data = np.array([id_to_vol[x] for x in X_train])
    X_val_data = np.array([id_to_vol[x] for x in X_val])
    #print(X_train_data.shape)
    #print(X_val_data.shape)
    print("-----------TRAIN/VALIDATION SPLIT COMPLETE-------------")
    train_loader = tf.data.Dataset.from_tensor_slices((X_train_data, Y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((X_val_data, Y_val))
    print("-----------TENSORFLOW DATA LOADING COMPLETE------------")
    #increase batch size
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

    print("-----------TENSORFLOW DATA PREPROCESSING COMPLETE--------")
    #data = train_dataset.take(1)
    #images, labels = list(data)[0]
    #images = images.numpy()
    #image = images[0]
    #print(image.shape)
    #plt.figure()
    #plt.imshow(image[:,:,7], cmap=plt.cm.gray)
    #plt.show()
    #print("-----------VISUALIZATION COMPLETE------------------------")

    model = ml.get_model()
    model.summary()
    print("-----------MODEL SUMMARY COMPLETE------------------------")

    # Compile model.
    initial_learning_rate = 0.0000001
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    #)
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),#lr_schedule),
        metrics=["acc"],
    )

        # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        #os.path.join("outputs/models/", '3d_synapse_classification_{epoch}') + datetime.now().strftime("%m_%d_%Y_%H:%M:%S") + ".h5"
        "outputs/models/3d_synapse_classification_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S") + ".h5" , save_best_only=True
    )
    #early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    # Train the model, doing validation at the end of each epoch
    epochs = 300
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb]#, early_stopping_cb],
    )

    print("-----------MODEL TRAINING COMPLETE------------------------")

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["acc", "loss"]):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])

    fig.savefig('outputs/models/model_performance_' + datetime.now().strftime("%m_%d_%Y_%H:%M:%S") + '.png')
    print("-----------MODEL PERFORMANCE VISUALIZATION COMPLETED------------------------")



if __name__ == "__main__":
    main()
# %%
#The 600 synapses
#df = pd.read_csv("data/dcvsyn600_annotated_12112022.csv")

# %%
#id_to_vol = ml.ingestor(df)

# %%
#viz.donut(al_bodies["cell_class"].value_counts().index.tolist(), 
#al_bodies["cell_class"].value_counts().values.tolist(), "Cell Class").write_image("figures/cell_class_distribution_flyaldump_"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")

#viz.donut(df["pre_syn_label"].value_counts().index.tolist(),
#df["pre_syn_label"].value_counts().values.tolist(), "Label").write_image("figures/syn600_label_distribution_"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")

# %%
'''
def questionable(row):
    if row["pre_syn_comment"] != "yes":
        return "questionable/no"
    else:
        return "yes"

df["questionable"] = df.apply(questionable, axis=1)
'''
# %%
#viz.donut(df["questionable"].value_counts().index.tolist(), 
#df["questionable"].value_counts().values.tolist(), "Synapse").write_image("figures/syn600_synlabel_distribution_"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")

# %%
#import tifffile
#vol1 = np.transpose(tifffile.imread("cutouts/dcvsyn1/img/vol.tiff"), (1,2,0))
#seg1 = np.transpose(tifffile.imread("cutouts/dcvsyn1/presyn/presyn.tiff"), (1,2,0))

# %%
#masked1 = (seg1/255)*vol1

# %%
'''def ingestor(df):
    id_to_vol = {}
    for index, row in df.iterrows():
        name = row["wk_id"]
        vol1 = np.transpose(tifffile.imread("cutouts/"+name+"/img/vol.tiff"), (1,2,0))
        seg1 = np.transpose(tifffile.imread("cutouts/"+name+"/presyn/presyn.tiff"), (1,2,0))
        masked1 = (seg1/255)*vol1
        id_to_vol[name] = masked1
    
    return id_to_vol
'''
# %%
#id_to_vol = ingestor(df)
#preprocessing steps: need to divide by 255
#plot any segmentation issues (outliars) and consider removing for training

# %%
#import tensorflow as tf 

# %%
'''
def get_model(width=160, height=160, depth=16):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=3, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)
'''