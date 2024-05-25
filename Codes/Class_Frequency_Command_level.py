import pandas as pd
import csv
import numpy as np

# This code calculates the class frequency based on the 'Label' column in your dataset of commands (.csv file)
def count_unique_labels(file_path):
    labels = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            label = int(row[-1])  # Assuming the last column is the label
            labels.append(label)

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_count_dict = dict(zip(unique_labels, label_counts))

    return label_count_dict

if __name__ == '__main__':
    file_path = 'Path to your dataset .csv file'
    label_counts = count_unique_labels(file_path)

    print("Label Counts:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count}")