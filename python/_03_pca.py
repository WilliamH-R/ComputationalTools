# %%
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text
from _02_clean import clean
from _99_functions import get_project_root

# %%
# Functions
def plot_pca_cat_w_loadings(pcs_df, pca, df, color, save_path, top_n=8, scale_loading=5):
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
    plt.title(f'PCA of Iris Dataset (PC1 vs PC2) with Top {top_n} Loadings', fontsize=16)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Add the most important loadings as arrows (vectors)
    features = df.columns
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

def plot_pca_cont_w_loadings(pcs_df, pca, df, color, save_path, top_n=8, scale_loading=5):
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
    plt.title(f'PCA of Iris Dataset (PC1 vs PC2) with Top {top_n} Loadings', fontsize=16)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Add the most important loadings as arrows (vectors)
    features = df.columns
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

# %%
# Set the project root folder
project_root = get_project_root()
os.chdir(project_root)

# Load data
df, _, non_bm = clean()

# Change label for TYPE from 0, 1 to Ovarian Cancer and Benign Ovarian Tumor
non_bm['TYPE'] = non_bm['TYPE'].replace({0: 'Benign Ovarian Tumor', 1: 'Ovarian Cancer'})

# Change label for menopause from 0, 1 to pre-menopause, post-menopause
non_bm['Menopause'] = non_bm['Menopause'].replace({0: 'Pre-menopause', 1: 'Post-menopause'})

# # Scale data
# df_scaled = StandardScaler().fit_transform(df)
# df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# # %%
# # Apply PCA
# pca = PCA()
# pcs_data = pca.fit_transform(df_scaled)

# # Convert the PCA results into a DataFrame
# pcs_df = pd.DataFrame(pcs_data, columns=[f'PC{i+1}' for i in range(pcs_data.shape[1])])

# # Add the non_bm data to the PCA DataFrame
# pcs_df = pcs_df.join(non_bm.reset_index(drop=True))

# #%%
# # 1. Plot cumulative explained variance
# explained_variance_ratio = pca.explained_variance_ratio_
# cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='b')
# plt.title('Cumulative Explained Variance by Principal Components', fontsize=16)
# plt.xlabel('Number of Principal Components', fontsize=14)
# plt.ylabel('Cumulative Explained Variance', fontsize=14)
# plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.ylim(0, 1.1)
# plt.grid(True)
# #plt.show()
# plt.savefig('results/cumulative_explained_variance.png')
# plt.close()

# # %%
# # 2. Plot PC1 vs PC2, colored by TYPE
# plot_pca_cat_w_loadings(pcs_df = pcs_df,
#                         pca = pca,
#                         df = df,
#                         color = 'TYPE',
#                         save_path = 'results/pc1_vs_pc2_type.png',
#                         top_n=8)

# # %%
# # 3. Plot PC1 vs PC2, colored by Menopause
# plot_pca_cat_w_loadings(pcs_df = pcs_df,
#                         pca = pca,
#                         df = df,
#                         color = 'Menopause',
#                         save_path = 'results/pc1_vs_pc2_menopause.png',
#                         top_n=8)

# # %%
# # 4. Plot PC1 vs PC2, colored by Age
# plot_pca_cont_w_loadings(pcs_df = pcs_df,
#                          pca = pca,
#                          df = df,
#                          color = 'Age',
#                          save_path = 'results/pc1_vs_pc2_age.png',
#                          top_n=8)

# # %%
# # 5. Plot the loadings of the PCA as barplot
# # Assuming your original features are in df.columns
# features = df.columns

# # Create a DataFrame for easier plotting and sorting (optional)
# loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# loading_lengths = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
# loading_df = pd.DataFrame({
#     'Feature': features,
#     'Loading Magnitude': loading_lengths
# }).sort_values(by='Loading Magnitude', ascending=False)

# # Plot the bar chart
# plt.figure(figsize=(10, 6))
# plt.barh(loading_df['Feature'], loading_df['Loading Magnitude'], color='skyblue')
# plt.xlabel('Loading Magnitude')
# plt.ylabel('Features')
# plt.title('Feature Contributions to PC1 and PC2')
# plt.gca().invert_yaxis()  # To have the largest loading at the top
# plt.tight_layout()

# # Save the plot or display it
# plt.savefig('results/pca_loadings_magnitude.png')
# plt.close()

# %%
##### ONLY MENOPAUSE DATA #####
# Subset 'df' to only menopause data
df_subset = df[non_bm['Menopause'] == 'Post-menopause']
non_bm_subset = non_bm[non_bm['Menopause'] == 'Post-menopause']

# Scale data
df_subset_scaled = StandardScaler().fit_transform(df_subset)
df_subset_scaled = pd.DataFrame(df_subset_scaled, columns=df.columns)

# %%
# Apply PCA
pca_subset = PCA()
pcs_data_subset = pca_subset.fit_transform(df_subset_scaled)

# Convert the PCA results into a DataFrame
pcs_df_subset = pd.DataFrame(pcs_data_subset, columns=[f'PC{i+1}' for i in range(pcs_data_subset.shape[1])])

# Add the non_bm_subset data to the PCA DataFrame
pcs_df_subset = pcs_df_subset.join(non_bm_subset.reset_index(drop=True))

#%%
# 1. Plot cumulative explained variance
explained_variance_ratio_subset = pca_subset.explained_variance_ratio_
cumulative_explained_variance_subset = np.cumsum(explained_variance_ratio_subset)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance_subset) + 1), cumulative_explained_variance_subset, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance by Principal Components', fontsize=16)
plt.xlabel('Number of Principal Components', fontsize=14)
plt.ylabel('Cumulative Explained Variance', fontsize=14)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.ylim(0, 1.1)
plt.grid(True)
#plt.show()
plt.savefig('results/cumulative_explained_variance_subset.png')
plt.close()

# %%
# 2. Plot PC1 vs PC2, colored by TYPE
plot_pca_cat_w_loadings(pcs_df = pcs_df_subset,
                        pca = pca_subset,
                        df = df,
                        color = 'TYPE',
                        save_path = 'results/pc1_vs_pc2_type_subset.png',
                        top_n=8)

# %%
# 3. Plot PC1 vs PC2, colored by Menopause
plot_pca_cat_w_loadings(pcs_df = pcs_df_subset,
                        pca = pca_subset,
                        df = df,
                        color = 'Menopause',
                        save_path = 'results/pc1_vs_pc2_menopause_subset.png',
                        top_n=8)

# %%
# 4. Plot PC1 vs PC2, colored by Age
plot_pca_cont_w_loadings(pcs_df = pcs_df_subset,
                         pca = pca_subset,
                         df = df,
                         color = 'Age',
                         save_path = 'results/pc1_vs_pc2_age_subset.png',
                         top_n=8)

# %%
# 5. Plot the loadings of the PCA as barplot
# Assuming your original features are in df.columns
features = df.columns

# Create a DataFrame for easier plotting and sorting (optional)
loadings_subset = pca_subset.components_.T * np.sqrt(pca_subset.explained_variance_)
loading_lengths_subset = np.sqrt(loadings_subset[:, 0]**2 + loadings_subset[:, 1]**2)
loading_df_subset = pd.DataFrame({
    'Feature': features,
    'Loading Magnitude': loading_lengths_subset
}).sort_values(by='Loading Magnitude', ascending=False)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.barh(loading_df_subset['Feature'], loading_df_subset['Loading Magnitude'], color='skyblue')
plt.xlabel('Loading Magnitude')
plt.ylabel('Features')
plt.title('Feature Contributions to PC1 and PC2')
plt.gca().invert_yaxis()  # To have the largest loading at the top
plt.tight_layout()

# Save the plot or display it
plt.savefig('results/pca_loadings_magnitude_subset.png')
plt.close()