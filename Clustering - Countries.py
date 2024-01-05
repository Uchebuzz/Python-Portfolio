#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from matplotlib.colors import Normalize
import seaborn as sns
import plotly.express as px


# In[9]:


from sklearn.cluster import KMeans, MeanShift


# In[10]:


co = pd.read_csv("./WLV dataset/country_data.csv")


# In[11]:


Figure= plt.figure(figsize= (10,15))
for col in co.columns:
    index = co.columns.get_loc(col)
    plt.subplot(4,5,index+1)
    plt.hist(co[col],bins = 10, color = "red")
    plt.title(f'{col}')
    plt.grid(True)
Figure.savefig("Histograms of countriescolumns.png")


# In[12]:


co.head()


# In[13]:


co.head()


# In[ ]:





# In[14]:


co.isnull().sum()


# In[15]:


co.info()


# In[16]:


X = co.iloc[:,[4,5]].values
# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 2, n_init='auto', random_state=5)
#model = MeanShift()
model.fit(X)


# In[17]:


cluster_centers = model.cluster_centers_
# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


# In[89]:


fig, ax = plt.subplots()
# store the normalisation of the color encodings
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)
# plot the clustered data
scatter1 = ax.scatter(X[:, 0], X[:, 1],
c = model.predict(X), s = 20, cmap = 'plasma', norm = nm)

ax.set_xlabel('Imports')
ax.set_ylabel('Income')


# In[ ]:





# In[90]:


X2 = co.iloc[:,[3,9]].values
# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 2, n_init='auto', random_state=4)
#model = MeanShift()
model.fit(X2)


# In[ ]:





# In[91]:


cluster_centers = model.cluster_centers_
# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


# In[92]:


fig, ax = plt.subplots()
# store the normalisation of the color encodings
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)
# plot the clustered data
scatter1 = ax.scatter(X2[:, 0], X2[:, 1],
c = model.predict(X), s = 50, cmap = 'plasma', norm = nm)

ax.set_xlabel('Income')
ax.set_ylabel('GDP')

Figure.savefig("Income and GDPP countriescolumns.png")


# In[ ]:





# In[94]:


X3 = co.iloc[:,[7,5]].values
# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 2, n_init='auto', random_state=5)
#model = MeanShift()
model.fit(X3)


# In[ ]:





# In[95]:


cluster_centers = model.cluster_centers_
# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


# In[96]:


fig, ax = plt.subplots()
# store the normalisation of the color encodings
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)
# plot the clustered data
scatter1 = ax.scatter(X3[:, 0], X3[:, 1],
c = model.predict(X3), s = 50, cmap = 'plasma', norm = nm)

ax.set_xlabel('Life Expentancy')
ax.set_ylabel('Income')

Figure.savefig("Life expentency and Income  countriescolumns.png")


# In[ ]:





# In[82]:


X4 = co.iloc[:,[6,7]].values
# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 2, n_init='auto', random_state=5)
#model = MeanShift()
model.fit(X4)


# In[83]:


cluster_centers = model.cluster_centers_
# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


# In[97]:


fig, ax = plt.subplots()
# store the normalisation of the color encodings
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)
# plot the clustered data
scatter1 = ax.scatter(X4[:, 0], X4[:, 1],
c = model.predict(X4), s = 50, cmap = 'plasma', norm = nm)

ax.set_xlabel('Life Expentancy')
ax.set_ylabel('Inflaton')

Figure.savefig("Life expentancy and Inflationcountriescolumns.png")


# #### 

# In[99]:


X5 = co.iloc[:,[2,4]].values
# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 3, n_init='auto', random_state=5)
#model = MeanShift()
model.fit(X5)


# In[100]:


cluster_centers = model.cluster_centers_
# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


# In[101]:


fig, ax = plt.subplots()
# store the normalisation of the color encodings
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)
# plot the clustered data
scatter1 = ax.scatter(X5[:, 0], X5[:, 1],
c = model.predict(X5), s = 50, cmap = 'plasma', norm = nm)

ax.set_xlabel('Exports')
ax.set_ylabel('Imports')

Figure.savefig("Exports and Imports countriescolumns.png")


# In[ ]:





# In[103]:


X6 = co.iloc[:,[3,7]].values
# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 2, n_init='auto', random_state=5)
#model = MeanShift()
model.fit(X6)


# In[104]:


cluster_centers = model.cluster_centers_
# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


# In[105]:


fig, ax = plt.subplots()
# store the normalisation of the color encodings
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)
# plot the clustered data
scatter1 = ax.scatter(X6[:, 0], X6[:, 1],
c = model.predict(X6), s = 50, cmap = 'plasma', norm = nm)

ax.set_xlabel('Health')
ax.set_ylabel('Life Expentancy')

Figure.savefig("Health and Life expentency countriescolumns.png")


# In[ ]:





# In[106]:


X7 = co.iloc[:,[3,1]].values
# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 3, n_init='auto', random_state=5)
#model = MeanShift()
model.fit(X7)


# In[107]:


cluster_centers = model.cluster_centers_
# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


# In[108]:


fig, ax = plt.subplots()
# store the normalisation of the color encodings
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)
# plot the clustered data
scatter1 = ax.scatter(X7[:, 0], X7[:, 1],
c = model.predict(X7), s = 50, cmap = 'plasma', norm = nm)

ax.set_xlabel('Health')
ax.set_ylabel('Child_Mort')

Figure.savefig("Health and child mortalityrams of countriescolumns.png")


# In[ ]:





# In[110]:


X8 = co.iloc[:,[9,1]].values
# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 3, n_init='auto', random_state=5)
#model = MeanShift()
model.fit(X8)


# In[111]:


cluster_centers = model.cluster_centers_
# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


# In[112]:


fig, ax = plt.subplots()
# store the normalisation of the color encodings
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)
# plot the clustered data
scatter1 = ax.scatter(X8[:, 0], X8[:, 1],
c = model.predict(X8), s = 50, cmap = 'plasma', norm = nm)

ax.set_xlabel('Child_mortaliy=ty')
ax.set_ylabel('GDPP')

Figure.savefig("GDPP and Child Mortalitycountriescolumns.png")


# In[ ]:





# In[37]:


#3d plot


# In[113]:


X9 = co.iloc[:,[9,1,3]].values
# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 3, n_init='auto', random_state=5)
#model = MeanShift()
model.fit(X9)


# In[114]:


cluster_centers = model.cluster_centers_
# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


# In[ ]:





# In[32]:


cluster_predictions = KMeans.predict(X)


# In[118]:


fig = plt.figure(figsize = (12,18))
ax = fig.add_subplot(111, projection = '3d' )
scatter1 = ax.scatter(X9[:, 0], X9[:, 1],X9[:, 2], c=model.predict(X9), s=50, cmap = 'plasma')
for i in range (centers.shape[0]):
    ax.text(centers[i,0], centers[i, 1], centers[i,2], str(i), c = "black", bbox = dict(boxstyle = 'round', facecolor = "white", edgecolor = 'black'))
    ax.azim = -60
    ax.dist = 10
    ax.elev = 10
    
ax.set_xlabel('GDPP')
ax.set_ylabel('Child_Mortality')
ax.set_zlabel('Health Care')
ax.set_title('KMeans Clustering in 3D')

Figure.savefig("3d countriescolumns.png")
    


# In[44]:


Co = co.drop("country", axis = 1)


# In[45]:


# Calculate correlation matrix
corr_matrix = Co.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Plot')
plt.show()


# In[ ]:




