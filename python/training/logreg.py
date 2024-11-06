#logistic regression for different dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from pathlib import Path

def get_project_root():
    """Returns the project root folder."""
    # Start from the current file's directory or notebook's location
    current_path = Path().resolve()

    # Loop to go up the directory tree until we reach the "ComputationalTools" folder
    while current_path.name != "ComputationalTools":
        if current_path.parent == current_path:  # We've reached the root directory without finding the folder
            raise FileNotFoundError("Could not find the 'ComputationalTools' folder in the directory hierarchy.")
        current_path = current_path.parent
        
    return current_path

# Set the project root folder
project_root = get_project_root()
os.chdir(project_root)

#Loading dataset in other way using dynamic pathing
df = pd.read_csv('data/cleaned.csv')
df_meta = pd.read_csv('data/non_bm.csv')[['TYPE']]

#Splitting the dataset into training and testing
df_train = df.iloc[:235]
df_test = df.iloc[235:]

#
df_meta_train = df_meta.iloc[:235]
df_meta_test  = df_meta.iloc[235:]

#import logistic regression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import linear_model


#Create a logistic regression classifier
logr = linear_model.LogisticRegression(class_weight='balanced')

logr.fit(df_train[['HE4','CEA']], df_meta_train['TYPE'])

# Predict the response for the test dataset
y_pred = logr.predict(df_test[['HE4', 'CEA']])

# Evaluate the model
accuracy = accuracy_score(df_meta_test['TYPE'], y_pred)
conf_matrix = confusion_matrix(df_meta_test['TYPE'], y_pred)
class_report = classification_report(df_meta_test['TYPE'], y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plotting the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate Sensitivity, Specificity, PPV, NPV, and Accuracy Rate
tn, fp, fn, tp = conf_matrix.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
accuracy_rate = (tp + tn) / (tp + tn + fp + fn)

print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Positive Predictive Value (PPV): {ppv}')
print(f'Negative Predictive Value (NPV): {npv}')
print(f'Accuracy Rate: {accuracy_rate}')


