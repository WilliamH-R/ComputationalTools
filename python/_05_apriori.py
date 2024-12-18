import pandas as pd
from mlxtend import frequent_patterns as fp
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import argparse

def main(split):
    ### Load data
    if split == "predefined":
        df_clean = pd.read_csv('data/preprocessed.csv')
        df_meta = pd.read_csv('data/non_bm.csv')[['TYPE']].astype(bool)

    elif split == "custom":
        df_clean = pd.read_csv('data/preprocessed_custom_split.csv')
        df_meta = pd.read_csv('data/non_bm_custom_split.csv')[['TYPE']].astype(bool)
        
    df_clean = df_clean.iloc[:235]
    df_clean = df_clean.apply(lambda x: np.where(x > np.percentile(x, 50), 1, 0)).astype(bool) # median transformation
    df_meta['TYPE'] = ~df_meta['TYPE'] # To ensure that cancer (0) is the active item
    df_meta  = df_meta.iloc[:235]
    
    # concatenate dataframes
    df = pd.concat([df_clean, df_meta], axis=1).astype(bool)

    # Apply apriori algorithm (get itemsets)
    df_fp = fp.apriori(df, min_support=0.0001, use_colnames=True,max_len=4)
    df_fp

    # Find association rules (compute association metrics, given itemsets)
    df_conf = fp.association_rules(df_fp, metric='confidence', min_threshold=0.001)
    # df_conf['interest'] = df_conf['confidence'] - df_conf['support']

    # only show consequents and antecedents that contain 'TYPE'
    top_rules = df_conf[df_conf['consequents'].apply(lambda x: 'TYPE' in x)].sort_values('lift', ascending=False)
    top_rules = top_rules[top_rules['confidence']<1][:25]

  
    print(top_rules)#[['antecedents', 'consequents', 'antecedent support', 'consequent support','support','confidence']]

    
    # Add nodes and edges
    all_nodes = []
    for _, row in top_rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                if consequent == 'TYPE':
                    all_nodes.append(antecedent)
        

    all_nodes = pd.Series(all_nodes)
    print(all_nodes.value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apriori Script")
    parser.add_argument('--split', type=str, required=True, help='The data split to use')
    
    args = parser.parse_args()
    main(args.split)