import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from keras.utils import to_categorical
import pandas as pd
import csv
import os
import random
from keras.optimizers import Adam
from keras.optimizers import Nadam
# Use the following commands if you run your code on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Use the following commands if you run your code on GPU. 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# define the focal loss function with gamma=2
def focal_loss(alpha_vec, gamma=2.0):
    alpha = tf.constant(alpha_vec, dtype=tf.float32)
    gamma = float(gamma)

    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1.e-9
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        #implement the equation of this loss function
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss_fixed

if __name__ == '__main__':
    file_path = 'path to your commands dataset'
    commands = []
    labels = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            command = [float(val) for val in row[:-1]]  # Convert command values to float
            label = int(row[-1])
            commands.append(command)
            labels.append(label)

    commands_array = np.array(commands, dtype=np.float32)
    labels_array = np.array(labels, dtype=float)
    labels_categorical = to_categorical(labels_array)

    # Calculate class weights
    unique_labels, label_counts = np.unique(labels_array, return_counts=True)
    total_samples = len(labels_array)
    num_classes = len(unique_labels)
    class_weights = {}
    for label, count in zip(unique_labels, label_counts):
        class_weights[label] = total_samples / (num_classes * count)
    
    # alpha vector is inversely proportional to the frequency of each class in the dataset
    alpha_vec = [class_weights[label] for label in sorted(unique_labels)]
    print("Class Weights:")
    for label, weight in class_weights.items():
        print(f"Class {label}: {weight}")
    print("Alpha Vector:")
    print(alpha_vec)
    loss = focal_loss(alpha_vec=alpha_vec, gamma=2.0)
   # MLP architecture 
    model = Sequential()
    model.add(Dense(256, input_dim=commands_array.shape[1], activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(labels_categorical.shape[1], activation='softmax'))

    model.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    model.fit(commands_array, labels_categorical, batch_size=32, epochs=100)
    model.save("your model name.h5")
    
