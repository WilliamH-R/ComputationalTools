#%%
# Imports
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Function for getting the project root folder
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

#path to save images
path_to_save = 'results/models'

# Load data frames
age_type_menopause = pd.read_csv('data/non_bm.csv')
data = pd.read_csv('data/cleaned.csv')
data = pd.concat([data, age_type_menopause[["Age", "Menopause"]]], axis=1)

# Get first 235 rows for training
train_df = data.iloc[:235, :]
train_labels_df = age_type_menopause["TYPE"].iloc[:235]

# Get last rows for testing
test_df = data.iloc[235:, :]
test_labels_df = age_type_menopause["TYPE"].iloc[235:]

# Specify the subsets of columns you want to use
subsets = [
    data.columns.tolist(),                                                # All features
    ["ALB", "HGB", "RBC", "TP"],                                          # Apriori subset
    ['ALB', 'HGB', 'HCT', 'LYM%', 'BASO%', 'PCT', 'TP', 'PLT'],           # PCA subset
    ["Menopause", "Age", "AFP", "CEA", "HE4", "CA19-9", "LYM%", "CO2CP"], # Article, 8 subset
    ["HE4", "CEA"]                                                        # Article, 2 subset
    # Add more subsets as needed
]

subset_names = [
    "All Features",
    "Apriori Subset",
    "PCA Subset",
    "Article 8 Subset",
    "Article 2 Subset"
    # Add more subset names as needed
]

# Get true labels as Series and encode them
train_labels = train_labels_df
test_labels = test_labels_df
n_classes = train_labels.nunique()

# Encode labels as integers (not necessary in binary, but helpful for consistency)
train_labels_encoded = (train_labels == 1).astype(int)
test_labels_encoded = (test_labels == 1).astype(int)

for i, subset in enumerate(subsets):
    # Select subset of columns for training and testing sets
    train_subset = train_df[subset]
    test_subset = test_df[subset]
    
    # Define a dictionary of models to evaluate
    models = {
        "K-Means": KMeans(n_clusters=2, random_state=42),
        "Logistic Regression": linear_model.LogisticRegression(class_weight='balanced'),
        "Regression Tree": DecisionTreeClassifier(random_state=42),
        "Hierarchical Clustering": AgglomerativeClustering(n_clusters=2, linkage='ward')
    }

    # Function to align cluster labels to true labels using the Hungarian algorithm
    def align_labels(true_labels, predicted_labels):
        cm = confusion_matrix(true_labels, predicted_labels)
        row_ind, col_ind = linear_sum_assignment(-cm)
        aligned_labels = np.zeros_like(predicted_labels)
        for k in range(2):
            aligned_labels[predicted_labels == col_ind[k]] = k
        return aligned_labels

    # Evaluate each model
    for model_name, model in models.items():
        # Fit the model
        model.fit(train_subset, train_labels_encoded)
        
        # Predict the response for the test dataset
        if model_name == "K-Means":
            y_pred = model.predict(test_subset)
            y_pred = align_labels(test_labels_encoded, y_pred)
        elif model_name == "Hierarchical Clustering":
            y_pred = model.fit_predict(test_subset)
            y_pred = align_labels(test_labels_encoded, y_pred)
        else:
            y_pred = model.predict(test_subset)
        
        # Evaluate the model
        accuracy = accuracy_score(test_labels_encoded, y_pred)
        conf_matrix = confusion_matrix(test_labels_encoded, y_pred)
        class_report = classification_report(test_labels_encoded, y_pred)
        
        print(f'Accuracy ({model_name}): {accuracy}')
        print(f'Confusion Matrix ({model_name}):')
        print(conf_matrix)
        print(f'Classification Report ({model_name}):')
        print(class_report)
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(test_labels_encoded, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{model_name} AUC = {roc_auc:.2f}')

    # Finalize the ROC plot
    plt.plot([0, 1], [0, 1], 'k--', lw=3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Subset {i+1} (Test Set)')
    plt.legend(loc="lower right")
    plt.savefig(f'{path_to_save}/roc_subset_{i+1}.png')
    plt.close()