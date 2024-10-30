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
from _02_clean import clean
from _99_functions import get_project_root

# Set the project root folder
project_root = get_project_root()
os.chdir(project_root)

# %%
# Load data
df, _, non_bm = clean()

# Change label for TYPE from 0, 1 to Ovarian Cancer and Benign Ovarian Tumor
non_bm['TYPE'] = non_bm['TYPE'].replace({0: 'Benign Ovarian Tumor', 1: 'Ovarian Cancer'})

# Change label for menopause from 0, 1 to pre-menopause, post-menopause
non_bm['Menopause'] = non_bm['Menopause'].replace({0: 'Pre-menopause', 1: 'Post-menopause'})

# Scale data
df_scaled = StandardScaler().fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled

# %%
# Apply PCA
pca = PCA()
pcs_data = pca.fit_transform(df_scaled)

# Convert the PCA results into a DataFrame
pcs_df = pd.DataFrame(pcs_data, columns=[f'PC{i+1}' for i in range(pcs_data.shape[1])])


# Add the non_bm data to the PCA DataFrame
pcs_df = pcs_df.join(non_bm.reset_index(drop=True))

#%%
# 1. Plot cumulative explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance by Principal Components', fontsize=16)
plt.xlabel('Number of Principal Components', fontsize=14)
plt.ylabel('Cumulative Explained Variance', fontsize=14)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.ylim(0, 1.1)
plt.grid(True)
#plt.show()
plt.savefig('results/cumulative_explained_variance.png')
plt.close()

# %%
# 2. Plot PC1 vs PC2, colored by TYPE
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pcs_df, x='PC1', y='PC2', hue='TYPE', palette='viridis', s=50)

# Set axis titles with explained variance
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
plt.title('PCA of Iris Dataset (PC1 vs PC2)', fontsize=16)
#plt.show()
plt.savefig('results/pc1_vs_pc2_type.png')
plt.close()

# %%
# 3. Plot PC1 vs PC2, colored by Menopause
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pcs_df, x='PC1', y='PC2', hue='Menopause', palette='viridis', s=50)

# Set axis titles with explained variance
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
plt.title('PCA of Iris Dataset (PC1 vs PC2)', fontsize=16)
#plt.show()
plt.savefig('results/pc1_vs_pc2_menopause.png')
plt.close()

# %%
# 4. Plot PC1 vs PC2, colored by Age
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pcs_df, x='PC1', y='PC2', hue='Age', palette='viridis', s=50)

# Set axis titles with explained variance
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
plt.title('PCA of Iris Dataset (PC1 vs PC2)', fontsize=16)
#plt.show()
plt.savefig('results/pc1_vs_pc2_age.png')
plt.close()

# %%
# 5. Plot PC1 vs PC2 with most important loadings (longest vectors)

# Get the PCA loadings (components) and calculate their vector lengths
loadings = pca.components_.T * np.sqrt(pca.explained_variance_) * 5
loading_lengths = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)

# Define the number of most important loadings to display (e.g., top 2)
top_n = 8
top_indices = np.argsort(loading_lengths)[-top_n:]  # Indices of the top N longest vectors

plt.figure(figsize=(8, 6))

# PC1 vs PC2 scatter plot
scatter = plt.scatter(pcs_df['PC1'], pcs_df['PC2'], c=pcs_df['Age'], cmap='viridis', s=50)

# Add color bar to represent the scale of Age
plt.colorbar(scatter, label='Age')

# Set axis titles with explained variance
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
plt.title(f'PCA of Iris Dataset (PC1 vs PC2) with Top {top_n} Loadings', fontsize=16)

# Add the most important loadings as arrows (vectors)
features = df.columns
for i in top_indices:
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5, head_width=0.05, head_length=0.05)
    plt.text(loadings[i, 0], loadings[i, 1], features[i], color='r', ha='center', va='center', fontsize=12)

# Save the plot with the top loadings
plt.grid(True)
#plt.show()
plt.savefig('results/pc1_vs_pc2_type_loadings.png')
plt.close()


# TO DO
# Plot each of the "meaningful" loadings in a separate barplot
# Plot pc1 vs pc2 colored by menopause
# Plot pc1 vs pc2 colored by age
# Consider making categories for age
# Make arrows thicker so they are more visible
# Text on arrow loadings should dodge
# Make PCA but only with menopause people