#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''importing the required librabries to perform data analysis'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[69]:


'''BASIC DATA SCIENCE TASK IN IRIS DATASET'''
#Analysis Iris DataSet
df=pd.read_csv('Iris.csv')
df


# In[8]:


#knowing the no of rows and columns in the taken dataset
df=df.reindex(np.random.permutation(df.index))
df


# In[15]:


#to view first 5 columns of the dataset we use head
df.head()


# In[14]:


#to use lasr 5 rows of data set we use tail
df.tail()


# In[17]:


df.dtypes


# In[18]:


df.shape


# In[19]:


df.info()


# In[20]:


df.describe()


# In[21]:


#total no of cells in the dataset
df.size


# In[84]:


'''MACHINE LEARNING ALGORITHM LOGISTIC REGRESSION USED'''
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets'''FOR MODEL EVALUATION'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a model (for example, Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

'''UTILIZING METRICS'''
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


# In[75]:


'''SIMPLE EXPLANATORY DATA ANALYSIS'''

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
target_names = iris.target_names

'''DISTRIBUTION OF EACH FEATURE IN DATASET'''
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()


# In[83]:


# Box plots for each feature by target class
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='target', y=feature, data=data)
    plt.title(f'Box Plot of {feature} by Target Class')

plt.tight_layout()
plt.show()


# In[82]:


# Pairplot to visualize relationships between features
sns.pairplot(data, hue='target', markers=["o", "s", "D"], palette='husl')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.tight_layout
plt.show()


# In[ ]:




