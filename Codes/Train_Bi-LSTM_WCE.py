import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import Masking
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Masking
from tensorflow.keras import regularizers
from keras.utils import to_categorical
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import pandas as pd
from sys import argv
import csv
import sys
import re
import os
import glob
import random
from keras.optimizers import Adam
from keras.optimizers import Nadam

# Use the following commands if you run your code on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#These command to run your code on GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate a subset of the total memory on each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#Implement the attention mechanism
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], input_shape[-1]), 
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape) 

    def call(self, x):
        e = K.tanh(K.dot(x, self.W)) # the attention scores (e) are calculated by applying the tanh activation function, and x is the encoded instance 
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# Define the Weighted Cross Categorical Entropy
def weighted_categorical_crossentropy(weights):    
    weights = tf.keras.backend.variable(weights)  

    def loss(y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.math.log(y_pred) * weights #WCE equation
        loss = -tf.reduce_sum(loss, -1)
        return loss

    return loss
 
if __name__ == '__main__': 
    files_path = 'path to your dataset folder of layers that are used for training'
    
    files=[]
    max_length=0
    max_file_name=None
    length_row=0
    count=0
    for filename in os.listdir(files_path):
     count+=1
     if filename.endswith(".csv"):
        
        file_path = os.path.join(files_path, filename)
        with open(file_path, 'r') as file:
            vectors = []
            csv_reader = csv.reader(file)
            next(csv_reader) 
            vectors.extend([row for row in csv_reader])
            for row in vectors:
             length_row=len(row)
            if len(vectors)>max_length: # Max length to pad all the layers to have the same elngth before feeding to the Bi-LSTM model
               max_length=len(vectors)
               max_file_name=filename
            files.append(vectors)
  
    print("max_length "+str(max_length))
    print("max_length filename "+str(max_file_name))
    print("length_row "+str(length_row))
    train_file_paths =os.listdir(files_path)
    train_labels_dic={}
    train_file_label=[]
    
     # Calculate the frequency of each class in the dataset based on the name of the layers
    count_fan, count_bed, count_ext, benign_count, count_layer = 0, 0, 0, 0 , 0
    count_cavity, count_density, count_state= 0, 0, 0
    for filename in train_file_paths:
        if "filament_cavity" in filename and  "malicious" in filename:
             count_cavity+=1
        elif "filament_density" in filename and  "malicious" in filename:
             count_density+=1           
        elif "filament_state" in filename and  "malicious" in filename:
             count_state+=1
        elif "fan_speed" in filename and  "malicious" in filename:
             count_fan+=1
        elif "bed_temperature" in filename and "malicious" in filename:
             count_bed+=1            
        elif "nozzle_temperature" in filename and  "malicious" in filename:
            count_ext+=1              
        elif "Z_profile" in filename  and  "malicious" in filename:
             count_layer+=1
        else:
             benign_count+=1


    
    
    for filename in train_file_paths:
        if "filament_cavity" in filename:           
             train_labels_dic[filename] =1 
             train_file_label.append((filename, 1))
        elif "filament_density" in filename:
             train_labels_dic[filename] =2 
             train_file_label.append((filename, 2))        
        elif "filament_state" in filename:  
             train_labels_dic[filename] =3 
             train_file_label.append((filename, 3))
        elif "fan_speed" in filename:           
             train_labels_dic[filename] =4 
             train_file_label.append((filename, 4))
        elif "bed_temperature" in filename:
             train_labels_dic[filename] =5 
             train_file_label.append((filename, 5))        
        elif "nozzle_temperature" in filename:  
             train_labels_dic[filename] =6 
             train_file_label.append((filename, 6))
        elif "LayerBonding" in filename or "layerBonding" in filename: 
             train_labels_dic[filename] =7 
             train_file_label.append((filename, 7))
        else:     
             train_labels_dic[filename] =0
             train_file_label.append((filename, 0)) 

    # Labeling the layers according to their names, as the malicious layers have "malicious" in their names along  with the attack type

    sequence_label=[]
    for filename in os.listdir(files_path):
     if filename.endswith(".csv"):
        file_path = os.path.join(files_path, filename)
        with open(file_path, 'r') as file:
            vectors = []
            padded_vectors=[]
            csv_reader = csv.reader(file)
            next(csv_reader) 
            vectors.extend([row for row in csv_reader])
            label=train_labels_dic[filename]
            if len(vectors) < max_length:  
                    pad_width = max_length - len(vectors)
                    padded_vectors = vectors + [[0.0]*length_row]*pad_width  # padding the layers to have the same length 
            else:    
                    padded_vectors= vectors
            sequence_label.append((padded_vectors,label))
    files_a= [t[0] for t in sequence_label]
    files_array = np.array(files_a,np.float32)
    train_labels = [t[1] for t in sequence_label]
    train_labels_array = np.array(train_labels,float)
    train_labels_categorical = to_categorical(train_labels_array)
    # Identify the architecture of the Bi-LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(256, input_shape=(max_length, length_row), return_sequences=True))) 
    model.add(Dropout(0.3))   
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3)) 
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3)) 
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3)) 
    model.add(AttentionLayer())
    model.add(Dense(train_labels_categorical.shape[1], activation="softmax"))
    opt = Nadam(learning_rate=0.001, clipvalue=1.0) 
    
    # Caluculate weights of the classes for WCE  function.
    # The weight is  inversely proportional to the frequency of each class in the dataset
    total_samples = benign_count + count_fan+ count_bed+ count_ext+ count_cavity+ count_density+ count_state+count_layer
    class_weights = {
    0: total_samples / (8 * benign_count),
    1: total_samples / (8 * count_cavity),
    2: total_samples / (8 * count_density),
    3: total_samples / (8 * count_state),
    4: total_samples / (8 * count_fan),
    5: total_samples / (8 * count_bed),
    6: total_samples / (8 * count_ext),
    7: total_samples / (8 *count_layer)
   
                   }
    print("Class Weights: ", class_weights)
    
    for a in class_weights:
        print(a)
    class_weights_array = np.array([class_weights[i] for i in range(len(class_weights))])
    loss = weighted_categorical_crossentropy(class_weights_array)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    files_reshaped = np.reshape(files_array, (files_array.shape[0], files_array.shape[1], files_array.shape[2]))
    labels_reshaped = np.reshape(train_labels_array,(train_labels_array.shape[0], 1))
    model.fit( files_reshaped,train_labels_categorical, batch_size=16, epochs=200)
    model.save("Bi-LSTM_WCE.h5") 
