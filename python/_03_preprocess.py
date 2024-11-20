import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

from _99_functions import get_project_root

def preprocess():
    # Set the project root folder
    project_root = get_project_root()
    os.chdir(project_root)

    # Read clean data
    df = pd.read_csv("data/cleaned.csv")

    # Logtransform specific columns
    cols_to_transform = ["ALP", "ALT", "AST", "CEA", "DBIL", "EO#", "EO%", "GGT", "HE4", "IBIL", "TBIL", "UA"]

    for col in cols_to_transform:
        # Ensure the column has no negative values before transforming
        min_value = df[df[col] > 0][col].min()

        if min_value < 0:
            raise ValueError(f"Column {col} contains negative values. Log transformation is not possible.")

        df[col] = np.log(df[col] + df[df[col] > 0][col].min() / 10)

    # Standardize all data
    df = (df - df.mean()) / df.std()

    # Save preprocessed df
    df.to_csv("data/preprocessed.csv", index=False)

    # Return preprocessed df
    return df


def new_split():
    # Load data sets
    df = pd.read_csv("data/preprocessed.csv")
    df_meta = pd.read_csv("data/non_bm.csv")

    # Gather data sets
    df_concat = pd.concat([df,df_meta],axis=1)

    df_meta1 = df_concat[["TYPE", "Age", "Menopause"]]

    # Split data sets into training and testing
    df2, df3 = train_test_split(df_concat, test_size=0.325, stratify=df_meta1["TYPE"], random_state=42)

    # Construct the meta data sets
    df_meta2 = df2[["TYPE", "Age", "Menopause"]]
    df_meta3 = df3[["TYPE", "Age", "Menopause"]]

    # Remove "TYPE", "Age", "Menopause" from the data sets
    df2 = df2.drop(columns=["TYPE", "Age", "Menopause"])
    df3 = df3.drop(columns=["TYPE", "Age", "Menopause"])

    # Save the data sets
    df_out = pd.concat([df2,df3],axis=0)
    df_meta_out = pd.concat([df_meta2,df_meta3],axis=0)

    df_out.to_csv("data/preprocessed_custom_split.csv", index=False)
    df_meta_out.to_csv("data/non_bm_custom_split.csv", index=False)

    return df_out, df_meta_out

if __name__ == "__main__":
    df_preprocessed = preprocess()
    df_out, df_meta_out = new_split()