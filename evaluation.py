# %%
import pandas as pd
from datetime import datetime
from neurometry import ml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pdb


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
    X_train, X_val, Y_train, Y_val = train_test_split(df["wk_id"], df["target"], test_size=.2, stratify=df["target"], random_state=49)
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    Y_train = np.array(Y_train)
    Y_val = np.array(Y_val)


    Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
    Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))

    X_train_data = np.array([id_to_vol[x] for x in X_train])
    X_val_data = np.array([id_to_vol[x] for x in X_val])
    print("-----------PRIOR MODEL LOAD START------------")
    model_one = ml.loader("outputs/models/3d_synapse_classification_01_27_2023_15:17:25.h5")
    
    #prediction = model_one.predict(np.expand_dims(x_val[0], axis=0))[0]
    
    #y_pred = model_one.predict(X_val_data, batch_size=2, verbose=1)

    #y_pred_alter = [int(x[0]>.5) for x in y_pred]
    #Y_val_alter = [x[0] for x in Y_val]

    #print(classification_report(Y_val_alter, y_pred_alter))

    pdb.set_trace()

    return




if __name__ == "__main__":
    main()