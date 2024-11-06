#%%
# Imports
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
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
    ["LYM%", "LYM#", "ALB", "HCT", "HGB", "IBIL", "TBIL", "DBIL"],        # PCA subset
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
    
    # Initialize and fit k-means on the training subset
    kmeans = KMeans(n_clusters=2, random_state=42)  # Use 2 clusters for binary
    kmeans.fit(train_subset)
    
    # Predict cluster labels on the test subset
    test_kmeans_labels = kmeans.predict(test_subset)
    
    # Align test cluster labels to true test labels using the Hungarian algorithm
    test_cm = confusion_matrix(test_labels_encoded, test_kmeans_labels)
    row_ind, col_ind = linear_sum_assignment(-test_cm)
    aligned_test_labels = np.zeros_like(test_kmeans_labels)
    for k in range(2):
        aligned_test_labels[test_kmeans_labels == col_ind[k]] = k
    
    # Confusion matrix after alignment
    aligned_test_cm = confusion_matrix(test_labels_encoded, aligned_test_labels)
    print(f"Confusion Matrix for Subset {i+1} (Test Set):")
    print(aligned_test_cm)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(test_labels_encoded, aligned_test_labels)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Subset {i+1} (Test Set)')
    plt.legend(loc="lower right")
    plt.show()