# Project 2
# Comp Selection Techniques for Residential Properties

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import sklearn 


df = pd.read_csv("housing_data.csv", header = 0)

df.columns.values[0] = "ROW_ID"
print(df.head())

# Shape of DataFrame
print(df.shape)

print("Missing Values")
# Missing Values 
print(df.isnull().sum())

# Correlation Plot 

corr = df.loc[:, ~df.columns.isin(['ROW_ID', 'PROP_ID'])].corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    horizontalalignment='right'
);

