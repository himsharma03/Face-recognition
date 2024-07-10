import os
from random import shuffle
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

encoding_file_path = './encoded-images-data.csv'
labels_fName = 'labels.pkl'
classifier_file = "./classifier.pkl"

if os.path.isfile(encoding_file_path):
    df = pd.read_csv(encoding_file_path)
else:
    print('\x1b[0;37;41m' + '{} does not exist'.format(encoding_file_path) + '\x1b[0m')
    quit()

if os.path.isfile(labels_fName):
    with open(labels_fName, 'rb') as f:
        le = pickle.load(f)
else:
    print('\x1b[0;37;41m' + '{} does not exist'.format(labels_fName) + '\x1b[0m')
    quit()

# Read the dataframe into a numpy array
# Shuffle the dataset
full_data = np.array(df.astype(float).values.tolist())
shuffle(full_data)

# Extract features and labels
X = np.array(full_data[:, 1:-1])
y = np.array(full_data[:, -1])

# Encode labels using LabelEncoder
le = LabelEncoder().fit(y)
y_encoded = le.transform(y)

# Initialize and train the classifier
clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
clf.fit(X, y_encoded)

# Validate training accuracy
train_predictions = clf.predict(X)
training_accuracy = np.mean(train_predictions == y_encoded)
print(f"Training Accuracy: {training_accuracy}")

# Backup existing classifier file
if os.path.isfile(classifier_file):
    backup_file = f"{classifier_file}.bak"
    if os.path.isfile(backup_file):
        os.remove(backup_file)  # Remove existing backup file
    os.rename(classifier_file, backup_file)

# Save the classifier to file
with open(classifier_file, 'wb') as f:
    pickle.dump((le, clf), f)
print('\x1b[6;30;42m' + f"Saving classifier to '{classifier_file}'" + '\x1b[0m')
