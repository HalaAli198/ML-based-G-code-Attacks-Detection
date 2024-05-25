import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import csv
import os
import random
# Use the following commands if you run your code on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Use these commands if you run your code on GPU. 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
# Load the train and test csv files.
def load_data(file_path):
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
    return np.array(commands, dtype=np.float32), np.array(labels, dtype=float)
# Calcualte the evaluationmetrics averaged across all classes (macro-averaging), as our dataset is imbalanced.
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

if __name__ == '__main__':
    train_file_path = 'path to your  commands train csv file'
    test_file_path = 'path to your  commands test csv file

    # Load the training dataset
    commands_train, labels_train = load_data(train_file_path)

    # Calculate class weights
    unique_labels, label_counts = np.unique(labels_train, return_counts=True)
    total_samples = len(labels_train)
    num_classes = len(unique_labels)
    class_weights = {}
    print("--------------Train----------------")
    for label, count in zip(unique_labels, label_counts):
        print("label: "+str(label) +"count "+str(count))
        class_weights[label] = total_samples / (num_classes * count)

    print("Class Weights:")
    for label, weight in class_weights.items():
        print(f"Class {label}: {weight}")

    # Train the Random Forest classifier with class weights
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
    rf_classifier.fit(commands_train, labels_train)

    # Load the test dataset
    commands_test, labels_test = load_data(test_file_path)
    unique_labels2, label_counts2 = np.unique(labels_test, return_counts=True)
    print("--------------Test----------------")
    for label, count in zip(unique_labels2, label_counts2):
        print("label: "+str(label) +"count "+str(count))
    # Evaluate RF on the test set
    y_pred = rf_classifier.predict(commands_test)

    # Calculate overall metrics
    accuracy, precision, recall, f1 = calculate_metrics(labels_test, y_pred)
    print(f"Overall Test Accuracy: {accuracy}")
    print(f"Overall Test Precision: {precision}")
    print(f"Overall Test Recall: {recall}")
    print(f"Overall Test F1-Score: {f1}")
    print("************************************************************")
    # Calculate metrics for each class
    conf_mat = confusion_matrix(labels_test, y_pred)
    class_metrics = calculate_class_metrics(conf_mat)
    print(conf_mat)
    print("************************************************************")
    print("Class-wise Metrics:")
    for i in range(num_classes):
        print(f"Class {i}:")
        print(f"Precision: {class_metrics['precision'][i]}")
        print(f"Recall: {class_metrics['recall'][i]}")
        print(f"F1-Score: {class_metrics['f1_score'][i]}")
        print("------------------------------------------------------------")
    # Calculate multiclass AUC
    y_pred_proba = rf_classifier.predict_proba(commands_test)
    auc_scores = calculate_multiclass_auc(labels_test, y_pred_proba, num_classes)
    print("Multiclass AUC Scores:")
    for label, auc in auc_scores.items():
        print(f"Class {label}: {auc}")
