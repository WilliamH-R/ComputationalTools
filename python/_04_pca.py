# %%
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text
from _99_functions import get_project_root
from _99_functions import plot_pca_cat_w_loadings
from _99_functions import plot_pca_cont_w_loadings

##### DECIDE ON WHICH DATA SPLIT TO USE #####
# Use either "predefined" or "custom"
split = "predefined"

# %%
# Set the project root folder
project_root = get_project_root()
os.chdir(project_root)

# Path to save images, either predfined split or custom split
if split == "predefined":
    path_to_save = 'results/pca_pre_split'
elif split == "custom":
    path_to_save = 'results/pca_custom_split'
else:
    print(f"'split' must be either 'predefined' or 'custom', not {split}")
    sys.exit(1)
os.makedirs(path_to_save, exist_ok=True)

### Load data
if split == "predefined":
    df = pd.read_csv('data/preprocessed.csv')
    age_type_menopause = pd.read_csv('data/non_bm.csv')

elif split == "custom":
    df = pd.read_csv('data/preprocessed_custom_split.csv')
    age_type_menopause = pd.read_csv('data/non_bm_custom_split.csv')


# Replace 0 and 1 with 'Cancer' and 'Benign' in the 'TYPE' column
age_type_menopause['TYPE'] = age_type_menopause['TYPE'].replace({0: 'Cancer', 1: 'Benign'})

# Standardize age
age_type_menopause['Age'] = (age_type_menopause['Age'] - age_type_menopause['Age'].mean()) / age_type_menopause['Age'].std()

# Add age such that it is used in PCA
df = pd.concat([df, age_type_menopause['Age']], axis=1)

# Take only first 235 rows for training
age_type_menopause = age_type_menopause.iloc[:235, :]
df = df.iloc[:235, :]

# Add menopause such that it is used in PCA
df_scaled = pd.concat([df, age_type_menopause['Menopause']], axis=1)

# Apply PCA
pca = PCA()
pcs_data = pca.fit_transform(df_scaled)

# Convert the PCA results into a DataFrame
pcs_df = pd.DataFrame(pcs_data, columns=[f'PC{i+1}' for i in range(pcs_data.shape[1])])

# Add age_type_menopause to use the columns for colouring
pcs_df = pd.concat([pcs_df, age_type_menopause], axis=1)

#%%
# 1. Plot cumulative explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='b')
# Wierdly enough, plt.suptitle is the main title
plt.suptitle('Cumulative Explained Variance by Principal Components', y=0.95, fontsize=16) # fiddle with y to ensure the two titles do not overlap
plt.title(f'Using {split} split', fontsize=10)
plt.xlabel('Number of Principal Components', fontsize=14)
plt.ylabel('Cumulative Explained Variance', fontsize=14)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.ylim(0, 1.1)
plt.grid(True)
#plt.show()
plt.savefig(f"{path_to_save}/cumulative_explained_variance.png")
plt.close()

# %%
# 2. Plot the loadings of the PCA as barplot
features = df_scaled.columns

# Create a DataFrame for easier plotting and sorting
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_lengths = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
loading_df = pd.DataFrame({
    'Feature': features,
    'Loading Magnitude': loading_lengths
}).sort_values(by='Loading Magnitude', ascending=False)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.barh(loading_df['Feature'], loading_df['Loading Magnitude'], color='skyblue')
plt.xlabel('Loading Magnitude')
plt.ylabel('Features')
# Wierdly enough, plt.suptitle is the main title
plt.title(f'Using {split} split', fontsize=10)
plt.suptitle('Feature Contributions to PC1 and PC2', y=0.95, fontsize=16) # fiddle with y to ensure the two titles do not overlap
plt.gca().invert_yaxis()  # To have the largest loading at the top
# plt.show()
plt.savefig(f"{path_to_save}/pca_loadings_magnitude.png")
plt.close()

# From the loadings magnitude plot, we see a drop after the first 8 features. Thus, we will only plot the top 8 loadings.
print("The top 8 features are:")
print(loading_df['Feature'][:8].tolist())

# %%
# 3. Plot PC1 vs PC2, colored by TYPE
plot_pca_cat_w_loadings(pcs_df = pcs_df,
                        pca = pca,
                        features = df_scaled.columns,
                        color = 'TYPE',
                        save_path = f"{path_to_save}/pc1_vs_pc2_type.png",
                        split = split,
                        top_n=8)

# %%
# 4. Plot PC1 vs PC2, colored by Menopause
plot_pca_cat_w_loadings(pcs_df = pcs_df,
                        pca = pca,
                        features = df_scaled.columns,
                        color = 'Menopause',
                        save_path = f"{path_to_save}/pc1_vs_pc2_menopause.png",
                        split = split,
                        top_n=8)

# %%
# 5. Plot PC1 vs PC2, colored by Age
plot_pca_cont_w_loadings(pcs_df = pcs_df,
                         pca = pca,
                         features = df_scaled.columns,
                         color = 'Age',
                         save_path = f"{path_to_save}/pc1_vs_pc2_age.png",
                         split = split,
                         top_n=8)

# # %%
# ##### ONLY MENOPAUSE DATA #####
# # Subset 'df' to only menopause data
# df_subset = df[age_type_menopause['Menopause'] == 'Post-menopause']
# age_type_menopause_subset = age_type_menopause[age_type_menopause['Menopause'] == 'Post-menopause']

# # Scale data
# df_subset_scaled = StandardScaler().fit_transform(df_subset)
# df_subset_scaled = pd.DataFrame(df_subset_scaled, columns=df.columns)

# # %%
# # Apply PCA
# pca_subset = PCA()
# pcs_data_subset = pca_subset.fit_transform(df_subset_scaled)

# # Convert the PCA results into a DataFrame
# pcs_df_subset = pd.DataFrame(pcs_data_subset, columns=[f'PC{i+1}' for i in range(pcs_data_subset.shape[1])])

# # Add the age_type_menopause_subset data to the PCA DataFrame
# pcs_df_subset = pcs_df_subset.join(age_type_menopause_subset.reset_index(drop=True))

# #%%
# # 1. Plot cumulative explained variance
# explained_variance_ratio_subset = pca_subset.explained_variance_ratio_
# cumulative_explained_variance_subset = np.cumsum(explained_variance_ratio_subset)

# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(cumulative_explained_variance_subset) + 1), cumulative_explained_variance_subset, marker='o', linestyle='--', color='b')
# plt.title('Cumulative Explained Variance by Principal Components', fontsize=16)
# plt.xlabel('Number of Principal Components', fontsize=14)
# plt.ylabel('Cumulative Explained Variance', fontsize=14)
# plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.ylim(0, 1.1)
# plt.grid(True)
# #plt.show()
# plt.savefig('results/cumulative_explained_variance_subset.png')
# plt.close()

# # %%
# # 2. Plot PC1 vs PC2, colored by TYPE
# plot_pca_cat_w_loadings(pcs_df = pcs_df_subset,
#                         pca = pca_subset,
#                         age_tfeatures = df.columnsype_menopause,
#                         color = 'TYPE',
#                         save_path = 'results/pc1_vs_pc2_type_subset.png',
#                         top_n=8)

# # %%
# # 3. Plot PC1 vs PC2, colored by Menopause
# plot_pca_cat_w_loadings(pcs_df = pcs_df_subset,
#                         pca = pca_subset,
#                         age_tfeatures = df.columnsype_menopause,
#                         color = 'Menopause',
#                         save_path = 'results/pc1_vs_pc2_menopause_subset.png',
#                         top_n=8)

# # %%
# # 4. Plot PC1 vs PC2, colored by Age
# plot_pca_cont_w_loadings(pcs_df = pcs_df_subset,
#                          pca = pca_subset,
#                          age_tfeatures = df.columnsype_menopause,
#                          color = 'Age',
#                          save_path = 'results/pc1_vs_pc2_age_subset.png',
#                          top_n=8)

# # %%
# # 5. Plot the loadings of the PCA as barplot
# # Assuming your original features are in df.columns
# features = df.columns

# # Create a DataFrame for easier plotting and sorting (optional)
# loadings_subset = pca_subset.components_.T * np.sqrt(pca_subset.explained_variance_)
# loading_lengths_subset = np.sqrt(loadings_subset[:, 0]**2 + loadings_subset[:, 1]**2)
# loading_df_subset = pd.DataFrame({
#     'Feature': features,
#     'Loading Magnitude': loading_lengths_subset
# }).sort_values(by='Loading Magnitude', ascending=False)

# # Plot the bar chart
# plt.figure(figsize=(10, 6))
# plt.barh(loading_df_subset['Feature'], loading_df_subset['Loading Magnitude'], color='skyblue')
# plt.xlabel('Loading Magnitude')
# plt.ylabel('Features')
# plt.title('Feature Contributions to PC1 and PC2')
# plt.gca().invert_yaxis()  # To have the largest loading at the top
# plt.tight_layout()

# # Save the plot or display it
# plt.savefig('results/pca_loadings_magnitude_subset.png')
# plt.close()