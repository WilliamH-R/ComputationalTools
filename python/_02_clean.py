"""
Data wrangling
"""
from _01_load import load
from IPython.display import display
import pandas as pd
from sklearn.impute import KNNImputer

import os
from _99_functions import get_project_root

def clean():

    # Set the project root folder
    project_root = get_project_root()
    os.chdir(project_root)

    data , bm_meta_data = load()

    ### Set the index of the data DataFrame to be the subject_ID
    data.set_index('SUBJECT_ID', inplace=True)
    # display(data.head())

    ### Count missing values in the data DataFrame
    missing_values_per_feature = data.isnull().sum().to_dict()
    # print("Missing values per feature:")
    # for feature, count in missing_values_per_feature.items():
    #     print(f"{feature}: {count}")


    ### Drop features with more than 7% missing (according to article)
    features_to_drop = []
    threshold = 0.07 * data.shape[0]
    for feature, count in missing_values_per_feature.items():
        if count > threshold:
            features_to_drop.append(feature)

    data.drop(features_to_drop, axis=1, inplace=True)
    # print(data.shape)
    # print("---------------------------\n")

    # Remove whitespace
    data.fillna('MISSING', inplace=True)
    data.replace(r'\s+', '', regex=True, inplace=True)  # Remove all whitespace characters (including tabs) from the data DataFrame
    data.replace('MISSING', pd.NA, inplace=True)        # Restore missing values to NaN

    # Hot-deck imputation for missing values
    data_numeric = data.apply(pd.to_numeric, errors='coerce') # Convert all non-numeric data to numeric
    imputer = KNNImputer(n_neighbors=5) 
    data_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data.columns, index=data.index)
    data = data_imputed

    #print("Missing values after hot-deck imputation:")
    #print(data.isnull().sum().to_dict())
    #print("---------------------------\n")

    # display(data.head())
    # print("---------------------------\n")

    ### Create new dataframes
    ### Find features in data that are not present in bm_meta_data
    non_bm_features = [col for col in data.columns if col not in bm_meta_data[bm_meta_data.columns[0]].values]
    # print(f"Features in data not present in bm_meta_data: {non_bm_features}")

    data_bm = data.drop(non_bm_features, axis=1) # axis = 1 means we are dropping columns

    data_non_bm = data[non_bm_features]

    # display(data_bm.head())
    # print("---------------------------\n")
    # display(data_non_bm.head())
    # print("---------------------------\n")
    """
    Data augmentation or something
    """
    ### Create a dictionary from the biomarker_abbrev DataFrame
    bm_dict = dict(zip(bm_meta_data[bm_meta_data.columns[0]], bm_meta_data[bm_meta_data.columns[1]]))

    ### Save cleaned df
    data_bm.to_csv('data/cleaned.csv', index=False)
    data_non_bm.to_csv('data/non_bm.csv', index=False)

    ### Return dataframe
    return data_bm, bm_dict, data_non_bm

if __name__ == "__main__":
    df_bm, bm_dict, df_non_bm = clean()

# Check if there is Nan values in df_preprocessed
print(df_bm.isna().sum().sum())