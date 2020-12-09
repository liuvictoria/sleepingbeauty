# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
helps tidy the following datasets:
gardner_time_to_catastrophe_dic_tidy.csv 
gardner_mt_catastrophe_only_tubulin.csv

functions:
    tidy_dic()
    tidy_concentrations()
    
Notes:
No inputs for either function
Keep the datasets in ../data file

"""


# +
import os
import pandas as pd
import numpy as np

data_path = "../data/"

# -

def tidy_dic():
    """
    tidy the gardner_time_to_catastrophe_dic_tidy.csv dataset
    reads the csv and adds a column that converts Boolean T/F to
    'labeled tubulin' vs 'microtubules'
    """
    #defining path for data
    fname = os.path.join(data_path, "gardner_time_to_catastrophe_dic_tidy.csv")

    #read csv
    df = pd.read_csv(fname)

    # Since just True or False on a plot legend doesn't make much sense, we'll create a column, 
    #```tubulin_labeled```, that converts the ```True``` and ```False``` values from the 
    #```labeled``` column to ```'labeled tubulin'``` and ```'microtubules'```
    df['tubulin_labeled'] = [
        'labeled tubulin' if df.labeled[i] else 'microtubules' 
        for i in range(len(df.labeled))
    ]
    return df

def tidy_concentrations(): 
    """
    tidy the gardner_mt_catastrophe_only_tubulin.csv dataset
    melts, removes nan, adds concentration_int columns
    """
    # defining path for data
    fname = os.path.join(data_path, "gardner_mt_catastrophe_only_tubulin.csv")

    df = pd.read_csv(fname, skiprows = 9)

    df = pd.melt(df, value_name='catastrophe time')
    df = df.rename(columns={"variable": "concentration"})

    df = df.dropna()

    #create new column to sort by (since 'concentration' column is a string rn)
    #pls don't delete this!
    df['concentration_int'] = np.array([
        int(uM_concentration[:-3])
        for uM_concentration in df.concentration.to_numpy()
    ])

    df = df.sort_values(by = ['concentration_int'])
    df = df.reset_index().drop(columns=['index'])
    return df

# +
#!jupytext --to python tidy_data.ipynb
