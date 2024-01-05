#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# In[53]:


# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score


# In[54]:


Hf = pd.read_csv("./WLV dataset/houseprice_data.csv")


# In[ ]:





# In[55]:


Figure= plt.figure(figsize= (20,15))
for col in Hf.columns:
    index = Hf.columns.get_loc(col)
    plt.subplot(4,5,index+1)
    plt.hist(Hf[col],bins = 10, color = "blue")
    plt.title(f'{col}')
    plt.grid(True)
Figure.savefig("Histograms of House pricing columns.png")


# In[56]:

# cleaning the dat and viewing
Hf.info()


# In[57]:


Hf.tail()


# In[58]:


Hf.isnull().sum()


# In[ ]:





# In[59]:


X = Hf.iloc[:, [2]].values # inputs Bathroom
y = Hf.iloc[:, 0].values # outputs price


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3,random_state=0)


# In[61]:


regr = LinearRegression()


# In[62]:


regr.fit(X_train, y_train)


# In[ ]:





# In[63]:


# visualise initial data set
fig1, ax = plt.subplots()
ax.scatter(X, y, color = 'green')
ax.plot(X_test, regr.predict(X_test), color = "red")
fig1.tight_layout()
#fig2.savefig('LR_initial_plot.png')


# Add labels to the x and y axes
plt.xlabel('Bathroom')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs Bathroom Initial Plot')


# In[ ]:





# In[64]:


# visualise predicted data set
fig2, ax = plt.subplots()
ax.scatter(X_test, y_test, color = 'green')
ax.plot(X_test, regr.predict(X_test), color = "red")


# Add labels to the x and y axes
plt.xlabel('Bathroom')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs Bathroom Predicted plot')



# In[65]:


#model2


# In[66]:


X2 = Hf.iloc[:, [1]].values # inputs bedrooms
y2 = Hf.iloc[:, 0].values # outputs price


# In[67]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size= 1/3,
random_state=0)


# In[68]:

# renaming our model
regr2 = LinearRegression()


# In[69]:


regr2.fit(X2_train, y2_train)


# In[70]:


# visualise initial data set
fig2, ax2 = plt.subplots()
ax2.scatter(X2, y2, color = 'green')
ax2.plot(X2_test, regr2.predict(X2_test), color = "red")
fig2.tight_layout()
#fig2.savefig('LR_initial_plot.png')


# In[ ]:





# In[71]:


# visualise predicted data set
fig2, ax2 = plt.subplots()
ax2.scatter(X2_test, y2_test, color = 'green')
ax2.plot(X2_test, regr2.predict(X2_test), color = "red")
fig2.tight_layout()
#fig2.savefig('LR_initial_plot.png')


# In[23]:


#model 3


# In[24]:


X3 = Hf.iloc[:, [12]].values # inputs year built
y3 = Hf.iloc[:, 0].values # outputs price


# In[25]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size= 1/3,
random_state=0)


# In[26]:


regr3 = LinearRegression()


# In[27]:


regr3.fit(X3_train, y3_train)


# In[28]:


# visualise initial data set
fig3, ax3 = plt.subplots()
ax3.scatter(X3, y3, color = 'green')
ax3.plot(X3_test, regr3.predict(X3_test), color = "red")
fig3.tight_layout()


# Add labels to the x and y axes
plt.xlabel('Year Built')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs Year Built Initial plot')
#fig2.savefig('LR_initial_plot.png')



# In[29]:


# visualise initial data set
fig4, ax3 = plt.subplots()
ax3.scatter(X3_test, y3_test, color = 'green')
ax3.plot(X3_test, regr3.predict(X3_test), color = "red")
fig4.tight_layout()


# Add labels to the x and y axes
plt.xlabel('Year Built')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs Year Built Predicted plot')
#fig2.savefig('LR_initial_plot.png')


# In[ ]:





# In[30]:


#model4


# In[31]:


X4 = Hf.iloc[:, [9]].values # inputs grade
y4 = Hf.iloc[:, 0].values # outputs price


# In[32]:


X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size= 1/3,
random_state=0)


# In[33]:


regr4 = LinearRegression()


# In[34]:


regr4.fit(X4_train, y4_train)


# In[72]:


# visualise initial data set
fig5, ax4 = plt.subplots()
ax4.scatter(X4, y4, color = 'green')
ax4.plot(X4_test, regr4.predict(X4_test), color = "red")
fig5.tight_layout()


# Add labels to the x and y axes
plt.xlabel('Grade')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs grade Initial plot')
#fig2.savefig('LR_initial_plot.png')


# In[ ]:





# In[73]:


# visualise predicted data set
fig6, ax4 = plt.subplots()
ax4.scatter(X4_test, y4_test, color = 'green')
ax4.plot(X4_test, regr4.predict(X4_test), color = "red")
fig6.tight_layout()


# Add labels to the x and y axes
plt.xlabel('Grade')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs Grade Predicted plot')
#fig2.savefig('LR_initial_plot.png')


# In[ ]:





# In[37]:


#model5


# In[ ]:





# In[38]:


X5 = Hf.iloc[:, [5]].values # inputs floors
y5 = Hf.iloc[:, 0].values # outputs price


# In[39]:


X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size= 1/3,
random_state=0)


# In[40]:


regr5 = LinearRegression()


# In[41]:


regr5.fit(X5_train, y5_train)


# In[42]:


# visualise initial data set
fig7, ax5 = plt.subplots()
ax5.scatter(X5, y5, color = 'green')
ax5.plot(X5_test, regr5.predict(X5_test), color = "red")
fig7.tight_layout()


# Add labels to the x and y axes
plt.xlabel('Floors')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs floors Initial plot')
#fig2.savefig('LR_initial_plot.png')


# In[43]:


fig6, ax5 = plt.subplots()
ax5.scatter(X5_test, y5_test, color = 'green')
ax5.plot(X5_test, regr5.predict(X5_test), color = "red")
fig6.tight_layout()


# Add labels to the x and y axes
plt.xlabel('floors')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs floors Predicted plot')
#fig2.savefig('LR_initial_plot.png')


# In[44]:


#Model6 - Square_feet


# In[45]:


X6 = Hf.iloc[:, [18]].values # inputs floors
y6 = Hf.iloc[:,0].values # outputs price


# In[46]:


X6_train, X6_test, y6_train, y6_test, = train_test_split(X6, y6, test_size= 1/3,random_state=0)


# In[47]:


regr6 = LinearRegression()


# In[48]:


regr6.fit(X6_train, y6_train)


# In[ ]:





# In[49]:


# visualise initial data set
fig9, ax6 = plt.subplots()
ax6.scatter(X6, y6, color = 'green')
ax5.plot(X6_test, regr6.predict(X6_test), color = "red")
fig9.tight_layout()


# Add labels to the x and y axes
plt.xlabel('Square_living')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs floors Initial plot')
#fig2.savefig('LR_initial_plot.png')


# In[ ]:





# In[50]:


fig10, ax6 = plt.subplots()
ax6.scatter(X6_test, y6_test, color = 'green')
ax6.plot(X6_test, regr6.predict(X6_test), color = "red")
fig6.tight_layout()


# Add labels to the x and y axes
plt.xlabel('squareliving')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs floors Predicted plot')
#fig2.savefig('LR_initial_plot.png')


# In[ ]:





# In[76]:


corr_matrix = Hf.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(16,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Plot')
plt.show()


# In[ ]:


#model 7


# In[78]:


X7 = Hf.iloc[:, [3]].values # inputs squarefoot living
y7 = Hf.iloc[:,0].values # outputs price


# In[79]:


X7_train, X7_test, y7_train, y7_test, = train_test_split(X7, y7, test_size= 1/3,random_state=0)


# In[81]:


regr7 = LinearRegression()


# In[82]:


regr7.fit(X7_train, y7_train)


# In[83]:


# visualise initial data set
fig10, ax7 = plt.subplots()
ax7.scatter(X7, y7, color = 'green')
ax7.plot(X7_test, regr7.predict(X7_test), color = "red")
fig9.tight_layout()


# Add labels to the x and y axes
plt.xlabel('Square_foot living')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs Square_foot living Initial plot')
#fig2.savefig('LR_initial_plot.png')


# In[ ]:





# In[84]:


#Predicted plot
fig10, ax7 = plt.subplots()
ax7.scatter(X7_test, y7_test, color = 'green')
ax7.plot(X7_test, regr7.predict(X7_test), color = "red")
fig7.tight_layout()


# Add labels to the x and y axes
plt.xlabel('squarefoot living')
plt.ylabel('Price')

# Add a title to the plot
plt.title('Price Vs floors Predicted plot')
#fig2.savefig('LR_initial_plot.png')


# ## 
