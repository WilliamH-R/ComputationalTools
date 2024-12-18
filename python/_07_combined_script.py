#%%
# Imports
import argparse

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

from scipy.optimize import linear_sum_assignment

from _99_functions import get_project_root

# Set the project root folder
project_root = get_project_root()
os.chdir(project_root)

def main(split):
    # Path to save images, either predfined split or custom split
    if split == "predefined":
        path_to_save = 'results/models_pre_split'
    elif split == "custom":
        path_to_save = 'results/models_custom_split'
    else:
        print(f"'--split' must be either 'predefined' or 'custom', not {split}")
        sys.exit(1)
    os.makedirs(path_to_save, exist_ok=True)

    ### Load data
    if split == "predefined":
        data = pd.read_csv('data/preprocessed.csv')
        age_type_menopause = pd.read_csv('data/non_bm.csv')
    elif split == "custom":
        data = pd.read_csv('data/preprocessed_custom_split.csv')
        age_type_menopause = pd.read_csv('data/non_bm_custom_split.csv')
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
        ['HE4', 'CA125', 'BASO%', 'TBIL', 'AST'],                             # Apriori Predefined Subset
        ['HE4', 'CA125', 'BASO%', 'TBIL', 'AST', 'CEA'],                      # Apriori Predefined Subset + CEA
        ['HE4', 'TBIL', 'CA125', 'BASO%', 'DBIL','IBIL'],                     # Apriori Custom Subset
        ['HE4', 'TBIL', 'CA125', 'BASO%', 'DBIL','IBIL', 'CEA'],              # Apriori Custom Subset + CEA
        ['ALB', 'HE4', 'HCT', 'BASO%', 'PLT', 'PCT', 'LYM%', 'HGB'],          # PCA Predefined Subset
        ['ALB', 'HE4', 'HCT', 'BASO%', 'PLT', 'PCT', 'LYM%', 'HGB', 'CEA'],   # PCA Predefined Subset + CEA
        ['HGB', 'HCT', 'ALB', 'IBIL', 'TBIL', 'HE4', 'LYM%', 'GLU.'],         # PCA Custom Subset
        ['HGB', 'HCT', 'ALB', 'IBIL', 'TBIL', 'HE4', 'LYM%', 'GLU.', 'CEA'],  # PCA Custom Subset + CEA
        ["Menopause", "Age", "AFP", "CEA", "HE4", "CA19-9", "LYM%", "CO2CP"], # Article, 8 subset
        ["HE4", "CEA"]                                                        # Article, 2 subset
        # Add more subsets as needed
    ]

    subset_names = [
        "All Features",
        "Apriori Predefined Subset",
        "Apriori Predefined Subset + CEA",
        "Apriori Custom Subset",
        "Apriori Custom Subset + CEA",
        "PCA Predefined Subset",
        "PCA Predefined Subset + CEA",
        "PCA Custom Subset",
        "PCA Custom Subset + CEA",
        "Article, subset of 8",
        "Article, subset of 2"
        # Add more subset names as needed
    ]

    subset_file_names = [
        "all_features",
        "apriori_pre",
        "apriori_pre_cea",
        "apriori_custom",
        "apriori_custom_cea",
        "pca_pre",
        "pca_pre_cea",
        "pca_custom",
        "pca_custom_cea",
        "subset_eight",
        "subset_two"
        # Add more subset names as needed
    ]

    # Get true labels as Series and encode them
    train_labels = train_labels_df
    test_labels = test_labels_df
    n_classes = train_labels.nunique()

    # Encode labels as integers (not necessary in binary, but helpful for consistency)
    train_labels_encoded = (train_labels == 1).astype(int)
    test_labels_encoded = (test_labels == 1).astype(int)

    # Function to align cluster labels to true labels using the Hungarian algorithm
    def align_labels(true_labels, predicted_labels):
        cm = confusion_matrix(true_labels, predicted_labels)
        _, col_ind = linear_sum_assignment(-cm)
        aligned_labels = np.zeros_like(predicted_labels)
        for k in range(2): 
            aligned_labels[predicted_labels == col_ind[k]] = k
        return aligned_labels

    for i, subset in enumerate(subsets):
        # Select subset of columns for training and testing sets
        train_subset = train_df[subset]
        test_subset = test_df[subset]
        
        # Define a dictionary of models to evaluate
        models = {
            "K-Means": KMeans(n_clusters=2, random_state=42),
            "Logistic Regression": linear_model.LogisticRegression(class_weight='balanced'),
            "Regression Tree": DecisionTreeClassifier(max_depth=2, random_state=42)
        }

        # Evaluate each model
        for model_name, model in models.items():
            # Fit the model
            model.fit(train_subset, train_labels_encoded)
            y_pred = model.predict(test_subset)
            
            # Predict the response for the test dataset
            if model_name == "K-Means":
                y_pred = align_labels(test_labels_encoded, y_pred)
            
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
            plt.plot(fpr, tpr, label=f'{model_name} = {roc_auc:.2f}')

        # Finalize the ROC plot
        plt.plot([0, 1], [0, 1], 'k--', lw=3)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.suptitle(f'ROC Curve for {subset_names[i]}', y=0.965, fontsize=16) # fiddle with y to ensure the two titles do not overlap
        plt.title(f'Using {split} split', fontsize=10)
        plt.legend(title="AUC", loc="lower right", title_fontproperties=font_manager.FontProperties(weight='bold'))
        plt.savefig(f'{path_to_save}/roc_{subset_file_names[i]}_{i+1}.png')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Script")
    parser.add_argument('--split', type=str, required=True, help='The data split to use')
    
    args = parser.parse_args()

    main(args.split)