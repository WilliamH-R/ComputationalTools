import pandas as pd
import numpy as np
import os

from _99_functions import get_project_root

def preprocess():
    # Set the project root folder
    project_root = get_project_root()
    os.chdir(project_root)

    # Read clean data
    df = pd.read_csv('data/cleaned.csv')

    # Logtransform specific columns
    cols_to_transform = ["ALP", "ALT", "AST", "CEA", "DBIL", "EO#", "EO%", "GGT", "HE4", "IBIL", "TBIL", "UA"]

    for col in cols_to_transform:
        df[col] = np.log(df[col] + np.min(df[df[col] != 0]) / 10)

    # Standardize all data
    df = (df - df.mean()) / df.std()

    # Save preprocessed df
    df.to_csv('data/preprocessed.csv', index=False)

    # Return preprocessed df
    return df

if __name__ == "__main__":
    df_preprocessed = preprocess()