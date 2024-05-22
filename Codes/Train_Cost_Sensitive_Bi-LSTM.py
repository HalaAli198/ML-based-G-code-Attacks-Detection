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

def create_class_weight(labels_dict, mu):
    #Create the cost items as [total/ class frequency]^gamma
    total = sum(labels_dict.values())
    class_weight = dict()
    print("total"+str(total))
    for key in labels_dict.keys():
        score = math.pow(total / float(labels_dict[key]), mu) 
        class_weight[key] = score

    return class_weight
 
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
            if len(vectors)>max_length:
               max_length=len(vectors)  # Max length to pad all the layers to have the same elngth before feeding to the Bi-LSTM model
               max_file_name=filename
            files.append(vectors)
  
    print("max_length "+str(max_length))
    print("max_length filename "+str(max_file_name))
    print("length_row "+str(length_row))
    train_file_paths =os.listdir(files_path)
    train_labels_dic={}
    train_file_label=[]
    
       
    # Labeling the layers according to their names, as the malicious layers have "malicious" in their names along  with the attack type
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
        elif "extruder_temperature" in filename:  
             train_labels_dic[filename] =6 
             train_file_label.append((filename, 6))
        elif "Z_profile" in filename: 
             train_labels_dic[filename] =7 
             train_file_label.append((filename, 7))
        else:     
             train_labels_dic[filename] =0
             train_file_label.append((filename, 0)) 


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
                    padded_vectors = vectors + [[0.0]*length_row]*pad_width # padding the layers to have the same length 
            else:    
                    padded_vectors= vectors
            sequence_label.append((padded_vectors,label))
    files_a= [t[0] for t in sequence_label]
    files_array = np.array(files_a,np.float32)
    train_labels = [t[1] for t in sequence_label]
    train_labels_array = np.array(train_labels,float)
    train_labels_categorical = to_categorical(train_labels_array)
    labels, counts = np.unique(train_labels_array, return_counts=True)
    labels_dict = dict(zip(labels, counts))
    # Calcualte the cost items , gamma in [0.1,0.9]. Best performance with 0.3
    class_weight = create_class_weight(labels_dict,0.3)
    print("Class Labels"+str(labels))
    print("Class Counts"+str(counts))
    print("Class Weights"+str(class_weight))
    
    # Identify the architecture of the Cost Sensitive Bi-LSTM model
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

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    files_reshaped = np.reshape(files_array, (files_array.shape[0], files_array.shape[1], files_array.shape[2]))
    labels_reshaped = np.reshape(train_labels_array,(train_labels_array.shape[0], 1))
    # incorporate the cost items into the backpropagation learning of Bi-LSTM
    model.fit( files_reshaped,train_labels_categorical, batch_size=16, epochs=200,class_weight=class_weight)
    model.save("Cost_Sensitive_Bi-LSTM.h5") 
    
