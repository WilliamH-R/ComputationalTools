"""
Loading
"""
# Python packages
import pandas as pd
from IPython.display import display

import os
from _99_functions import get_project_root

def load():

    # Set the project root folder
    project_root = get_project_root()
    os.chdir(project_root)
    
    ### Define the file paths
    file_path_1 = '_raw/Supplementary data 1.xlsx'
    file_path_2 = '_raw/Supplementary data 2.xlsx'

    ### Load data
    data = pd.read_excel(file_path_1)
    # print(f"Dimensions of data dataframe: {data.shape}")
    bm_meta_data = pd.read_excel(file_path_2)
    # print(f"Dimensions of biomarker meta data dataframe: {bm_meta_data.shape}")

    ### display data
    # display(data.head())
    # print("---------------------------\n")
    # display(bm_meta_data.head())
    # print("---------------------------\n")
    """
    Data exploration
    """
    ### We need equal amount of type 0 and type 1 entries
    ### for median not to be messed with
    type_zero_count = (data['TYPE'] == 0).sum()
    type_one_count = (data['TYPE'] == 1).sum()
    # print(f"Number of entries where type is 0: {type_zero_count}")
    # print(f"Number of entries where type is 1: {type_one_count}")
    # print("---------------------------\n")

    ### Save raw
    data.to_csv("data/raw.csv", index=False)
    bm_meta_data.to_csv("data/meta.csv", index=False)

    ### Return df
    return data, bm_meta_data

