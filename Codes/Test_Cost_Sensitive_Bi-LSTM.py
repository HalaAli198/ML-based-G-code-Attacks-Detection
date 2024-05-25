import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import csv
import os
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, precision_score, recall_score, roc_auc_score  
import re
# Use the following commands if you run your code on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#These command to run your code on GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
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


# Calcualte the evaluation metrics averaged across all classes (macro-averaging), as our dataset is imbalanced.
def calculate_metrics(y_true, y_pred, average_type='macro'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average_type, labels=np.unique(y_true))
    recall = recall_score(y_true, y_pred, average=average_type, labels=np.unique(y_true))
    f1 = f1_score(y_true, y_pred, average=average_type, labels=np.unique(y_true))
    return accuracy, precision, recall, f1

def calculate_multiclass_auc(y_true, y_pred, num_classes):
    auc_scores = {}
    y_true_array = np.array(y_true)  
    y_pred_array = np.array(y_pred)  
    for i in range(num_classes):
        class_true = (y_true_array == i).astype(int)
        class_pred = (y_pred_array == i).astype(int)
        if len(np.unique(class_true)) > 1:  
            auc_scores[i] = roc_auc_score(class_true, class_pred)
        else:
            auc_scores[i] = float('nan') 
    return auc_scores

def calculate_class_metrics(conf_mat):
    precision = []
    recall = []
    f1 = []

    for i in range(len(conf_mat)):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        TN = conf_mat.sum() - (TP + FP + FN)

        prec = TP / (TP + FP) if (TP + FP) != 0 else 0
        rec = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)

    return {'precision': precision, 'recall': recall, 'f1_score': f1}

# Get the actual labels of the layers
def get_actual_class(filename):
    
    if "filament_cavity" in filename and "malicious" in filename:
        return 1
    elif "filament_density" in filename and "malicious" in filename:
        return 2
    elif "filament_state" in filename and "malicious" in filename:
        return 3
    elif "fan_speed" in filename and "malicious" in filename:
        return 4
    elif "bed_temperature" in filename and "malicious" in filename:
        return 5
    elif "nozzle_temperature" in filename and "malicious" in filename:
        return 6
    elif "Z_profile" in filename  and  "malicious" in filename:
        
         return 7
             
    else:
        return 0  


def class_wise_accuracy(confusion_matrix):
        class_accuracy = {}
        for i, row in enumerate(confusion_matrix):
            class_accuracy[i] = row[i] / row.sum() if row.sum() > 0 else 0
        return class_accuracy

if __name__ == '__main__':
    files_dir = 'path to your dataset folder of layers that are used for testing'

    sequence_label = []
    file_layers = defaultdict(list)
    layer_actual = []
    layer_predictions = []
    # as the sequences must have the same sequence and as the max length is the max length between the train and test dataset.
    # Calcualte the max length in each dataset and takes the max between them. 
    max_length=9799 #9865 # 9920, 9892
    max_file_name=None
    length_row=0
    for filename in os.listdir(files_dir):
     if filename.endswith(".csv"): 
        file_path = os.path.join(files_dir, filename)
        with open(file_path, 'r') as file:
            vectors = []
            csv_reader = csv.reader(file)
            next(csv_reader)  
            vectors.extend([row for row in csv_reader])
            for row in vectors:
             length_row=len(row)
            if len(vectors)>max_length:
                print(filename)
                
               
    
    for filename in os.listdir(files_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(files_dir, filename)
            with open(file_path, 'r') as file:
                vectors = []
                csv_reader = csv.reader(file)
                next(csv_reader)  
                vectors.extend([row for row in csv_reader])
                if len(vectors) < max_length:
                    padded_vectors = vectors + [[0.0] * len(vectors[0])] * (max_length - len(vectors))  # padding the layers to have the same length 
                else:    
                    padded_vectors= vectors
                sequence_label.append((filename, np.array(padded_vectors, np.float32)))

            common_identifier = extract_common_identifier(filename)
            file_layers[common_identifier].append(filename)
            layer_actual.append(get_actual_class(filename))
    files_array = np.array([t[1] for t in sequence_label])
    files_reshaped = np.reshape(files_array, (files_array.shape[0], files_array.shape[1], files_array.shape[2]))
    model = load_model('Cost_Sensitive_Bi-LSTM.h5', custom_objects={'AttentionLayer': AttentionLayer})
    

    #predict the labels of the layers in the test dataset folder
    pred_labels = model.predict(files_reshaped, verbose=0)
    predicted_classes = np.argmax(pred_labels, axis=1)
    layer_predictions.extend(predicted_classes)


    # Calculate accuracy
    layer_accuracy = accuracy_score(layer_actual, layer_predictions)
    layer_confusion_mat = confusion_matrix(layer_actual, layer_predictions)
    layer_class_accuracy = class_wise_accuracy(layer_confusion_mat)
    layer_f1_macro = f1_score(layer_actual, layer_predictions, average='macro')
    print("Layer-level Accuracy: ", layer_accuracy)
    print("Layer-level F1 Macro Score: ", layer_f1_macro)
    print("Layer-level Confusion Matrix:")
    print("------------------------------------")
    layer_accuracy_macro, layer_precision_macro, layer_recall_macro, layer_f1_macro = calculate_metrics(layer_actual, layer_predictions, average_type='macro')
    print(" Macro Layer-level Accuracy: ", layer_accuracy_macro)
    print("Macro Layer-level Precision: ", layer_precision_macro)
    print("Macro Layer-level Recall: ", layer_recall_macro)
    print("Macro Layer-level F1 Score: ", layer_f1_macro)
    print(layer_confusion_mat)
    print("Layer-level Per-Class Accuracy: ", layer_class_accuracy)
    layer_class_metrics = calculate_class_metrics(layer_confusion_mat)
    print("Layer-level Per-Class Precision: ", layer_class_metrics['precision'])
    print("Layer-level Per-Class Recall: ", layer_class_metrics['recall'])
    print("Layer-level Per-Class F1 Score: ", layer_class_metrics['f1_score'])
    print("******************************************************************************************************************************")
    model.summary()
    model_config = model.get_config()
    print(model_config)
