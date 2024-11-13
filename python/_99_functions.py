# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from adjustText import adjust_text

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

def plot_pca_cat_w_loadings(pcs_df, pca, features, color, save_path, split, top_n=8, scale_loading=5):
    # To get the variance explained on axis
    explained_variance_ratio = pca.explained_variance_ratio_

    # Get the PCA loadings (components) and calculate their vector lengths
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_) * scale_loading
    loading_lengths = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)

    # Take the number of most important loadings to display 
    top_indices = np.argsort(loading_lengths)[-top_n:]

    plt.figure(figsize=(8, 6))

    # PC1 vs PC2 scatter plot
    sns.scatterplot(data=pcs_df, x='PC1', y='PC2', hue=color, palette='viridis', s=50)

    # Set axis titles with explained variance
    plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
    plt.suptitle(f'PCA of Iris Dataset (PC1 vs PC2) with Top {top_n} Loadings', y=0.95, fontsize=16) # fiddle with y to ensure the two titles do not overlap
    plt.title(f'Using {split} split', fontsize=10)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Add the most important loadings as arrows (vectors)
    texts = []
    for i in top_indices:
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', width = 0.05, alpha=0.5, head_width=0.05, head_length=0.05)
        text = plt.text(loadings[i, 0], loadings[i, 1], features[i], color='darkred', ha='center', va='center')
        texts.append(text)

    # Automatically adjust text to avoid overlap
    adjust_text(texts, only_move={'points': 'xy', 'texts': 'xy'}, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))


    # Save the plot with the top loadings
    plt.grid(True)
    #plt.show()
    plt.savefig(save_path)
    plt.close()

def plot_pca_cont_w_loadings(pcs_df, pca, features, color, save_path, split, top_n=8, scale_loading=5):
    # To get the variance explained on axis
    explained_variance_ratio = pca.explained_variance_ratio_

    # Get the PCA loadings (components) and calculate their vector lengths
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_) * scale_loading
    loading_lengths = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)

    # Take the number of most important loadings to display 
    top_indices = np.argsort(loading_lengths)[-top_n:]

    plt.figure(figsize=(8, 6))

    # PC1 vs PC2 scatter plot
    scatter = plt.scatter(pcs_df['PC1'], pcs_df['PC2'], c=pcs_df[color], cmap='viridis', s=50)

    # Add color bar to represent the scale of 'color'
    plt.colorbar(scatter, label=color)

    # Set axis titles with explained variance
    plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
    plt.suptitle(f'PCA of Iris Dataset (PC1 vs PC2) with Top {top_n} Loadings', y=0.95, fontsize=16) # fiddle with y to ensure the two titles do not overlap
    plt.title(f'Using {split} split', fontsize=10)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Add the most important loadings as arrows (vectors)
    texts = []
    for i in top_indices:
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', width = 0.05, alpha=0.5, head_width=0.05, head_length=0.05)
        text = plt.text(loadings[i, 0], loadings[i, 1], features[i], color='darkred', ha='center', va='center')
        texts.append(text)

    # Automatically adjust text to avoid overlap
    adjust_text(texts, only_move={'points': 'xy', 'texts': 'xy'}, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Save the plot with the top loadings
    plt.grid(True)
    #plt.show()
    plt.savefig(save_path)
    plt.close()