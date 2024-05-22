import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, precision_score, recall_score, roc_auc_score  
import csv
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
# define the focal loss function with gamma=2
def focal_loss(alpha_vec, gamma=2.0):
    alpha = tf.constant(alpha_vec, dtype=tf.float32)
    gamma = float(gamma)

    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1.e-9
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        #implement the equation of this loss function
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed
# Calcualte the evaluationmetrics averaged across all classes (macro-averaging), as our dataset is imbalanced.
def calculate_metrics(y_true, y_pred, average_type='macro'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average_type, labels=np.unique(y_true))
    recall = recall_score(y_true, y_pred, average=average_type, labels=np.unique(y_true))
    f1 = f1_score(y_true, y_pred, average=average_type, labels=np.unique(y_true))
    return accuracy, precision, recall, f1


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
def extract_common_identifier(filename):
    return re.sub(r'(_\d+)?(_malicious)?\.csv', '', filename)


def class_wise_accuracy(confusion_matrix):
        class_accuracy = {}
        for i, row in enumerate(confusion_matrix):
            class_accuracy[i] = row[i] / row.sum() if row.sum() > 0 else 0
        return class_accuracy


if __name__ == '__main__':
    file_path = 'path to your commands test dataset.csv'

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
    test_commands_array = np.array(commands, dtype=np.float32)
    test_labels_array = np.array(labels, dtype=float)
    test_labels_categorical = to_categorical(test_labels_array)
    #This alpha vector should be the same as the one used during the training phase.
    #By using the same alpha vector, you ensure that the class weights are applied consistently when calculating the focal loss during testing.
    alpha_vec =   [0.23549206036653117, 6.9952997702109885, 3.86836481256859, 5.690627920808905, 5.66943198171506]      
    loss = focal_loss(alpha_vec=alpha_vec, gamma=2.0)
    # Load the trained model
    model = load_model(" your trained model name.h5", custom_objects={'focal_loss_fixed': focal_loss(alpha_vec)})  
    # Evaluate the model on the test set
    y_pred = model.predict(test_commands_array)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(test_labels_categorical, axis=1)
    

    command_accuracy = accuracy_score(y_test_classes, y_pred_classes)
    command_confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)
    command_class_accuracy = class_wise_accuracy(command_confusion_mat)
    command_f1_macro = f1_score(y_test_classes, y_pred_classes, average='macro')
    print("Command-level Accuracy: ", command_accuracy)
    print("Command-level F1 Macro Score: ", command_f1_macro)
    print("Command-level Confusion Matrix:")
    print("------------------------------------")
    command_accuracy_macro, command_precision_macro, command_recall_macro, command_f1_macro = calculate_metrics(y_test_classes, y_pred_classes, average_type='macro')
    print(" Macro Command-level Accuracy: ", command_accuracy_macro)
    print("Macro Command-level Precision: ", command_precision_macro)
    print("Macro Command-level Recall: ", command_recall_macro)
    print("Macro Command-level F1 Score: ", command_f1_macro)
    print(command_confusion_mat)
    print("command-level Per-Class Accuracy: ", command_class_accuracy)
    command_class_metrics = calculate_class_metrics(command_confusion_mat)
    print("command-level Per-Class Precision: ", command_class_metrics['precision'])
    print("commandlevel Per-Class Recall: ", command_class_metrics['recall'])
    print("commandlevel Per-Class F1 Score: ", command_class_metrics['f1_score'])
    print("******************************************************************************************************************************")

    
    # Print the classification report
    report = classification_report(y_test_classes, y_pred_classes, digits=4)
    print("Classification Report:")
    print(report)
    print("Model Summary:")
    model.summary()
