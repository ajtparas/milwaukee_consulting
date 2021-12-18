#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np


# In[15]:


df = pd.read_csv("imputeddatafinal.csv")
df.head(5)


# In[16]:


df.shape


# In[17]:


df.isnull().sum()


# In[18]:


df.info


# In[19]:


# Inspect the categorical variables
df.select_dtypes('object').nunique()


# In[20]:


df.describe()


# In[21]:


df = df.drop(['PROP_ID', 'NBHD', 'SALE_DATE', 'SALE_PRICE', "APPEALED19", 'APPEALED20', 'APPEALED21'], axis = 1)


# In[22]:


one_hot = pd.get_dummies(df['BLD_TYPE'])
df = df.drop('BLD_TYPE',axis = 1)
df = df.merge(one_hot, how='outer', left_index=True, right_index=True)

one_hot = pd.get_dummies(df['APPRAISER'])
df = df.drop('APPRAISER',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['QUAL'])
df = df.drop('QUAL',axis = 1)
df = df.merge(one_hot, how='outer', left_index=True, right_index=True)

one_hot = pd.get_dummies(df['COND'])
df = df.drop('COND',axis = 1)
df = df.merge(one_hot, how='outer', left_index=True, right_index=True)

one_hot = pd.get_dummies(df['KITCHEN_RATING'])
df = df.drop('KITCHEN_RATING',axis = 1)
df = df.merge(one_hot, how='outer', left_index=True, right_index=True)

one_hot = pd.get_dummies(df['FULL_BATH_RATING'])
df = df.drop('FULL_BATH_RATING',axis = 1)
df = df.join(one_hot, how='outer')

one_hot = pd.get_dummies(df['HALF_BATH_RATING'])
df = df.drop('HALF_BATH_RATING',axis = 1)
df = df.merge(one_hot, how='outer', left_index=True, right_index=True)


# In[23]:


df.shape


# In[24]:


from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(df)  
transformer
Normalizer()
tdf = transformer.transform(df)
df = pd.DataFrame(tdf, columns = df.columns)
df


# In[25]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(df)
pca_df = pca.transform(df)
pca_df


# In[26]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[27]:


sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(pca_df)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[28]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(pca_df)


# In[29]:


df = pd.read_csv("imputeddatafinal.csv")


# In[30]:


df_final = pd.concat([df.reset_index(drop =True), pd.DataFrame(pca_df)], axis = 1)


# In[31]:


df_final.columns.values[-2: ] = ['Component_1', 'Component_2']


# In[32]:


df_final['Cluster'] = kmeans.labels_


# In[33]:


df_final.shape


# In[34]:


x_axis = df_final['Component_1'] 
y_axis = df_final['Component_2']
plt.figure(figsize = (10,8))
import seaborn as sns
sns.scatterplot(x_axis, y_axis, hue = df_final['Cluster'], palette = ['g', 'c', 'm'])
plt.title('Clusters')
plt.show()


# In[35]:


df_Cluster_0 = df_final.loc[df_final['Cluster'] == 0]
df_Cluster_1 = df_final.loc[df_final['Cluster'] == 1]
df_Cluster_2 = df_final.loc[df_final['Cluster'] == 2]


# In[36]:


from scipy import spatial

df_kd_0 = df_Cluster_0[['Component_1', 'Component_2']]
df_kd_1 = df_Cluster_1[['Component_1', 'Component_2']]
df_kd_2 = df_Cluster_2[['Component_1', 'Component_2']]


# In[47]:


tree = spatial.cKDTree(df_kd_1)

# Replace PROP_ID With A property ID number below
distances, indices = tree.query(df_kd_1.loc[PROP_ID].values, k=1+1)
similar_properties = df_kd_1.iloc[indices[1:]].assign(Distance=distances[1:])

print(similar_properties)


# In[51]:


df_final.to_csv('Final.csv', index=False)

