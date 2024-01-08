#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[116]:


# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB 
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap, Normalize


# In[117]:


Rf = pd.read_csv("./WLV dataset/nba_rookie_data.csv")


#Cleaning the data


Rf.head()


# In[119]:


Rf.info()


# In[120]:


Rookie = Rf.drop("Name", axis = 1)


# In[ ]:





# In[121]:


Figure= plt.figure(figsize= (20,10))
for col in Rookie.columns:
    index = Rookie.columns.get_loc(col)
    plt.subplot(4,5,index+1)
    plt.hist(Rookie[col],bins = 10, color = "blue")
    plt.title(f'{col}')
    plt.grid(True)
Figure.savefig("Histograms of columns.png")


# In[122]:


Rookie.corr()


# In[123]:


corr_matrix = Rookie.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(16, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Plot')
plt.show()


# In[124]:


#logistic


# In[125]:


X1 = Rookie[['Games Played']]
y1 = Rookie['TARGET_5Yrs']


# In[ ]:





# In[126]:


# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)


# In[127]:


# Initialize the logistic regression model
model = LogisticRegression()
# Train the model
model.fit(X1_train, y1_train)


# In[ ]:





# In[ ]:





# In[128]:


# output the accuracy score
print('Our Accuracy is %.2f' % model.score(X1_test, y1_test))
# output the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (X1_test.shape[0], (y1_test != model.predict(X1_test)).sum()))


# In[129]:


# Make predictions on the test set
y1_pred = model.predict(X1_test)


# In[161]:


print("Classification Report:")
print(classification_report(y1_test, y1_pred))

print("Confusion Matrix:")
print(confusion_matrix(y1_test, y1_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y1_test, y1_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Logistic regression Confusion Matrix')
plt.show()


# In[131]:


plt.scatter(X1_train, y1_train, color='blue', label='Training data')
plt.scatter(X1_test, y1_test, color='red', label='Testing data')

# Creating the decision boundary
x_values = np.linspace(X1.min(), X1.max(), 100)
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]  # Probability of class 1

# Plotting the decision boundary
plt.plot(x_values, y_values, color='green', label='Decision boundary')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.title('Logistic Regression - Decision Boundary')
plt.show()


# In[ ]:





# In[132]:


#GNB


# In[133]:


X2 = Rookie[['Games Played']]
y2 = Rookie['TARGET_5Yrs']


# In[134]:


# split the data into training and test sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size= 1/3, random_state=0)


# In[135]:


gnb = GaussianNB()
gnb.fit(X2_train, y2_train)


# In[ ]:





# In[136]:


print('Number of mislabeled points out of a total of %d points: %d'
% (X2_test.shape[0], (y2_test != gnb.predict(X2_test)).sum()))

print('Our accuracy ', gnb.score(X2_test, y2_test))
y2_pred = gnb.predict(X2_test)
print('Predict a value:', y2_pred)


# In[162]:


print("Classification Report:")
print(classification_report(y2_test, y2_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y2_test, y2_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('GNB Confusion Matrix')
plt.show()


# In[ ]:





# In[138]:


plt.figure(figsize=(8, 6))
sns.histplot(data=Rookie, x="Games Played", hue='TARGET_5Yrs', element='step', kde=True)
plt.xlabel("Games Played")
plt.ylabel('Count')
plt.title(f'Distribution of Classes based on {"Games Played"}')
plt.legend(title='TARGET_5Yrs')
plt.show()


# In[ ]:





# In[165]:


X_2 = Rookie[['Minutes Played']]
y_2 = Rookie['TARGET_5Yrs']


# In[166]:


# split the data into training and test sets
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size= 1/3, random_state=0)


# In[167]:


gnb = GaussianNB()
gnb.fit(X_2_train, y_2_train)


# In[168]:


print('Number of mislabeled points out of a total of %d points: %d'
% (X_2_test.shape[0], (y_2_test != gnb.predict(X_2_test)).sum()))

print('Our accuracy ', gnb.score(X_2_test, y_2_test))
y_2_pred = gnb.predict(X_2_test)
print('Predict a value:', y_2_pred)


# In[169]:


print("Classification Report:")
print(classification_report(y_2_test, y_2_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_2_test, y_2_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('GNB Confusion Matrix')
plt.show()


# In[180]:


plt.figure(figsize=(8, 6))
sns.histplot(data=Rookie, x="Minutes Played", hue='TARGET_5Yrs', element='step', kde=True)
plt.xlabel("Minutes Played")
plt.ylabel('Count')
plt.title(f'Distribution of Classes based on {"Minutes Played"}')
plt.legend(title='TARGET_5Yrs')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[163]:


#Neural_Network


# In[183]:


X3 = Rookie[['Games Played']]
y3 = Rookie['TARGET_5Yrs']


# In[184]:


# split the data into training and test sets
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size= 1/3, random_state=0)


# In[ ]:





# In[185]:


model = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500)  # Adjust parameters as needed
model.fit(X3_train, y3_train)


# In[186]:


# Make predictions
predictions = model.predict(X3_test)


# In[190]:


accuracy = accuracy_score(y3_test, predictions)
print(f"Accuracy: {accuracy}")

print('Number of mislabeled points out of a total of %d points: %d'
% (X3_test.shape[0], (y3_test != model.predict(X3_test)).sum()))


# In[188]:


print("Classification Report:")
print(classification_report(y3_test, predictions))

print("Confusion Matrix:")
cm = confusion_matrix(y3_test, predictions)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Neural Network Confusion Matrix')
plt.show()


# In[179]:


# Define the range for the plot
x_min, x_max = X3_train['Games Played'].min() - 1, X3_train['Games Played'].max() + 1

# Create a meshgrid to generate a visualization
xx = np.arange(x_min, x_max, 0.1).reshape(-1, 1)  # Use only one feature for prediction

# Make predictions across the entire space
predictions = model.predict(xx)

# Plot decision boundaries
plt.plot(xx, predictions, color='blue')
plt.scatter(X3_train['Games Played'], y3_train, c=y3_train, cmap=plt.cm.RdYlBu, edgecolor='k')
plt.xlabel('Games Played')
plt.ylabel('TARGET_5yrs')
plt.title('Neural Network Decision Boundaries')
plt.show()

