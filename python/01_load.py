"""
Loading
"""
# Python packages
import pandas as pd
import os
from IPython.display import display
from sklearn.impute import KNNImputer

# Define the file paths
file_path_1 = './_raw/Supplementary data 1.xlsx'
file_path_2 = './_raw/Supplementary data 2.xlsx'

# Load data
data = pd.read_excel(file_path_1)
print(f"Dimensions of data dataframe: {data.shape}")
bm_meta_data = pd.read_excel(file_path_2)
print(f"Dimensions of biomarker meta data dataframe: {bm_meta_data.shape}")

# display data
display(data.head())
print("---------------------------\n")
display(bm_meta_data.head())
print("---------------------------\n")
"""
Data exploration
"""
# We need equal amount of type 0 and type 1 entries
### for median not to be messed with
type_zero_count = (data['TYPE'] == 0).sum()
type_one_count = (data['TYPE'] == 1).sum()
print(f"Number of entries where type is 0: {type_zero_count}")
print(f"Number of entries where type is 1: {type_one_count}")
print("---------------------------\n")
"""
Data wrangling
"""
# Set the index of the data DataFrame to be the subject_ID
data.set_index('SUBJECT_ID', inplace=True)
display(data.head())

# Count missing values in the data DataFrame
missing_values_per_feature = data.isnull().sum().to_dict()
print("Missing values per feature:")
for feature, count in missing_values_per_feature.items():
    print(f"{feature}: {count}")


# Drop features with more than 7% missing (according to article)
features_to_drop = []
threshold = 0.07 * data.shape[0]
for feature, count in missing_values_per_feature.items():
    if count > threshold:
        features_to_drop.append(feature)

data.drop(features_to_drop, axis=1, inplace=True)
print(data.shape)
print("---------------------------\n")

# Remove whitespace
data.fillna('MISSING', inplace=True)
data.replace(r'\s+', '', regex=True, inplace=True)  # Remove all whitespace characters (including tabs) from the data DataFrame
data.replace('MISSING', pd.NA, inplace=True)        # Restore missing values to NaN

# Hot-deck imputation for missing values
data_numeric = data.apply(pd.to_numeric, errors='coerce') # Convert all non-numeric data to numeric
imputer = KNNImputer(n_neighbors=5) 
data_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data.columns, index=data.index)
data = data_imputed

print("Missing values after hot-deck imputation:")
print(data.isnull().sum().to_dict())
print("---------------------------\n")

display(data.head())
print("---------------------------\n")

# Create new dataframes
### Find features in data that are not present in bm_meta_data
non_bm_features = [col for col in data.columns if col not in bm_meta_data[bm_meta_data.columns[0]].values]
print(f"Features in data not present in bm_meta_data: {non_bm_features}")

data_bm = data.drop(non_bm_features, axis=1) # axis = 1 means we are dropping columns

data_non_bm = data[non_bm_features]

display(data_bm.head())
print("---------------------------\n")
display(data_non_bm.head())
print("---------------------------\n")
"""
Data augmentation or something
"""
# Create a dictionary from the biomarker_abbrev DataFrame
bm_dict = dict(zip(bm_meta_data[bm_meta_data.columns[0]], bm_meta_data[bm_meta_data.columns[1]]))