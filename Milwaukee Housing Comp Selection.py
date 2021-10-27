#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd
import numpy as np


# In[142]:


df = pd.read_csv("housing_data.csv")
df.head(5)


# In[143]:


df.shape


# In[144]:


df.isnull().sum()


# In[145]:


df['HALF_BATH_CT'] = df['HALF_BATH_CT'].fillna(0)


# In[146]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(df[['FULL_BATH_CT', 'FINISHED_AREA', 'LAND_SF']])

imputed_df = df.copy()
imputed_df[['FULL_BATH_CT', 'FINISHED_AREA', 'LAND_SF']] = imputer.transform(df[['FULL_BATH_CT', 'FINISHED_AREA', 'LAND_SF']])

df = pd.DataFrame(imputed_df)
df


# In[147]:


imp = SimpleImputer(strategy="most_frequent")

imputer = imp.fit(df[['QUAL', 'COND', 'KITCHEN_RATING', 'FULL_BATH_RATING', 'HALF_BATH_RATING']])

imputed_df = df.copy()
imputed_df[['QUAL', 'COND', 'KITCHEN_RATING', 'FULL_BATH_RATING', 'HALF_BATH_RATING']] = imputer.transform(df[['QUAL', 'COND', 'KITCHEN_RATING', 'FULL_BATH_RATING', 'HALF_BATH_RATING']])

df = pd.DataFrame(imputed_df)
df


# In[148]:


df["HALF_BATH_RATING"].value_counts()


# In[149]:


# converting type of columns to 'category'and assigning numerical values and storing in another column
df['BLD_TYPE'] = df['BLD_TYPE'].astype('category')
df['BLD_TYPE'] = df['BLD_TYPE'].cat.codes
df['APPRAISER'] = df['APPRAISER'].astype('category')
df['APPRAISER'] = df['APPRAISER'].cat.codes
df['NBHD'] = df['NBHD'].astype('category')
df['NBHD'] = df['NBHD'].cat.codes
df['QUAL'] = df['QUAL'].astype('category')
df['QUAL'] = df['QUAL'].cat.codes
df['COND'] = df['COND'].astype('category')
df['COND'] = df['COND'].cat.codes
df['KITCHEN_RATING'] = df['KITCHEN_RATING'].astype('category')
df['KITCHEN_RATING'] = df['KITCHEN_RATING'].cat.codes
df['FULL_BATH_RATING'] = df['FULL_BATH_RATING'].astype('category')
df['FULL_BATH_RATING'] = df['FULL_BATH_RATING'].cat.codes
df['HALF_BATH_RATING'] = df['HALF_BATH_RATING'].astype('category')
df['HALF_BATH_RATING'] = df['HALF_BATH_RATING'].cat.codes
df


# In[150]:


df = df.drop(["Unnamed: 0", "PROP_ID", "SALE_DATE", "SALE_PRICE", "APPEALED19", "APPEALED20", "APPEALED21"], axis = 1)
df


# In[153]:


from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(df)  
transformer
Normalizer()
tdf = transformer.transform(df)
df = pd.DataFrame(tdf, columns = df.columns)
df


# In[154]:


from sklearn.decomposition import PCA
df = pd.DataFrame(data=np.random.normal(0, 1, (20, 10)))

pca = PCA(n_components=2)
pca.fit(df)

pca.components_ 


# In[40]:


from sklearn.cluster import KMeans


# In[41]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(df)


# In[ ]:


# Plan 
# For Numerical Variables with NA Values - Random Forest Imputation
# For Categorical Variables with NA Values - MissForest Imputation
# Encode Categorical Variables - Label Encoding?
# Standardize/Normalize 
# PCA - Reduce Dimensions 
# K-Means Clustering (Maybe Try Out Other Methods Too)
# K-Means Based Reccomendation System

