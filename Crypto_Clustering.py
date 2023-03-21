#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
import holoviews as hv
hv.extension('bokeh', 'matplotlib')


# In[2]:


# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    "Resources/crypto_market_data.csv",
    index_col="coin_id")

# Display sample data
df_market_data.head(10)


# In[3]:


# Generate summary statistics
df_market_data.describe()


# In[4]:


# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Prepare the Data

# In[7]:


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)


# In[8]:


# Create a DataFrame with the scaled data

df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()


# ---

# ### Find the Best Value for k Using the Original Data.

# In[13]:


# Create a list with the number of k-values from 1 to 11
k = range(1,12)
list(k)


# In[15]:


# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list

for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_market_data_scaled)
    inertia.append(model.inertia_)

inertia


# In[16]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the Elbow curve
elbow_data_df = pd.DataFrame(elbow_data)


# In[17]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot = elbow_data_df.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
elbow_plot


# #### Answer the following question: 
# 
# **Question:** What is the best value for `k`?
# 
# **Answer:** The best value is 4 for 'K'.

# ---

# ### Cluster Cryptocurrencies with K-means Using the Original Data

# In[18]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=1)


# In[19]:


# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)


# In[21]:


# Predict the clusters to group the cryptocurrencies using the scaled data
crypto_clusters = model.predict(df_market_data_scaled)

# Print the resulting array of cluster values.
print(crypto_clusters)


# In[23]:


# Create a copy of the DataFrame
df_crypto_pred = df_market_data_scaled.copy()


# In[24]:


# Add a new column to the DataFrame with the predicted clusters
df_crypto_pred["CryptoCluster"] = crypto_clusters

# Display sample data
df_crypto_pred.head()


# In[28]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.

pred_plot = df_crypto_pred.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="CryptoCluster",
    hover_cols=['coin_id'],
    title = "Scatter Plot by Crypto Segment"
)
pred_plot


# ---

# ### Optimize Clusters with Principal Component Analysis.

# In[30]:


# Create a PCA model instance and set `n_components=3`.

pca = PCA(n_components=3)


# In[32]:


# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
market_data_pca = pca.fit_transform(df_market_data_scaled)
# View the first five rows of the DataFrame. 
market_data_pca[:5]


# In[33]:


# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_


# In[34]:


pca.explained_variance_ratio_.sum()


# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** 89.5%

# In[40]:


# Create a new DataFrame with the PCA data.

market_pca_df = pd.DataFrame(market_data_pca, columns=["PC1", "PC2", "PC3"])


# Copy the crypto names from the original data
market_pca_df["coin_id"] = df_market_data.index

# Set the coinid column as index
market_pca_df = market_pca_df.set_index("coin_id")

# Display sample data
market_pca_df.head()


# ---

# ### Find the Best Value for k Using the PCA Data

# In[36]:


# Create a list with the number of k-values from 1 to 11
k = list(range(1,12))
k


# In[42]:


# Create an empy list to store the inertia values
# Create an empy list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list

for i in k:
    model = KMeans(n_clusters = i)
    model.fit(market_pca_df)
    inertia.append(model.inertia_)

inertia


# In[43]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data_pca = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data_pca)
df_elbow_pca


# In[44]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot_pca = df_elbow_pca.hvplot.line(x="k", y="inertia", title="Elbow Curve Using PCA Data", xticks=k)
elbow_plot_pca


# #### Answer the following questions: 
# 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:**
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** 

# ### Cluster Cryptocurrencies with K-means Using the PCA Data

# In[45]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)


# In[46]:


# Fit the K-Means model using the PCA data
model.fit(market_pca_df)


# In[47]:


# Predict the clusters to group the cryptocurrencies using the PCA data
crypto_clusters_pca = model.predict(market_pca_df)
# Print the resulting array of cluster values.
print(crypto_clusters_pca) 


# In[48]:


# Create a copy of the DataFrame with the PCA data
market_pca_predictions = market_pca_df.copy()

# Add a new column to the DataFrame with the predicted clusters
market_pca_predictions["CryptoCluster"] = crypto_clusters_pca

# Display sample data
market_pca_predictions.head()


# In[49]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.

pred_plot_pca = market_pca_predictions.hvplot.scatter(
    x="PC1",
    y="PC2",
    by="CryptoCluster",
    title = "Scatter Plot by Stock Segment - PCA=3"
)
pred_plot_pca


# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

# In[50]:


# Composite plot to contrast the Elbow curves

elbow_plot + elbow_plot_pca


# In[51]:


# Composite plot to contrast the clusters
pred_plot + pred_plot_pca


# #### Answer the following question: 
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:** The utilization of PCA to reduce the features into three principal components for clustering the data results in a more widespread distribution of the data and a clearer distinction between clusters. Despite some outliers, both techniques exhibit comparable scatter and effectively define separate clusters. However, the advantage of the PCA approach lies in its ability to convey cluster patterns with greater clarity, as it integrates more data despite utilizing fewer features.

# In[ ]:




