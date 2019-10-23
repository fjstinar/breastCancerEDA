#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


# In[7]:


dataFrame = pd.read_csv('data.csv')


# In[15]:


yDiag = dataFrame.diagnosis
xFeats = dataFrame.drop(['Unnamed: 32', 'id', 'diagnosis'], axis = 1)


# Now that I have the data in a usable state I will make sure that the data says what I expect it to so I will see the number of benign and malignant cancerous patients.

# In[19]:


ax = sns.countplot(yDiag)


# Thus we have an idea of what this dataset says. There are approximately double the number of patients with benign tumors than those with malignant tumors.
# 
# Now I will begin trying to visualize the data so that I can pick out some important features. Before doing this though I will look at the actually data to see if the data needs to be standardized or to see if it is fine as it is.

# In[20]:


xFeats.describe()


# Actually I don't really think so - there are a lot of features though so one plot would make it hard to see all of them so I will split it up. I will standardize the data (I took this standardization code from the DATAI team in Turkey). I am choosing to use box plots here so we can see outliers.

# In[27]:


data = xFeats
#first 10 features
data_n_2 = (data - data.mean()) / (data.std())              
data = pd.concat([yDiag,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')

plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue = "diagnosis", data=data)
plt.xticks(rotation=90)


# In[30]:


#next 10 features
data = pd.concat([yDiag,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')

plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue = "diagnosis", data=data)
plt.xticks(rotation=90)


# In[29]:


#last 10 features
data = pd.concat([yDiag,data_n_2.iloc[:,20:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')

plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue = "diagnosis", data=data)
plt.xticks(rotation=90)


# To better see a correlation between certain features and certain classifications I read online that a good way to do this is through a swarm plot for each feature. I will lay out the swarm plots in the same way in which I laid out the box plots.

# In[36]:


#first 10
data_d = yDiag
data = xFeats
data_n2 = (data - data.mean()) / (data.std())
data = pd.concat([yDiag,data_n2.iloc[:,0:10]],axis = 1)
data = pd.melt(data,id_vars = "diagnosis",
              var_name = "features",
              value_name = "value")
plt.figure(figsize=(10,10))
tick = time.time()
sns.swarmplot(x="features", y = "value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# In[39]:


#next 10
data = pd.concat([yDiag,data_n2.iloc[:,10:20]],axis = 1)
data = pd.melt(data,id_vars = "diagnosis",
              var_name = "features",
              value_name = "value")
plt.figure(figsize=(10,10))
tick = time.time()
sns.swarmplot(x="features", y = "value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# In[40]:


#last 10
data = pd.concat([yDiag,data_n2.iloc[:,20:31]],axis = 1)
data = pd.melt(data,id_vars = "diagnosis",
              var_name = "features",
              value_name = "value")
plt.figure(figsize=(10,10))
tick = time.time()
sns.swarmplot(x="features", y = "value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# I intially started to select features from the swarm plots based on the features that had the most clear difference between the malignant and benign classifications. I first picked out area_mean and perimeter_mean as both of them seemed to have a strong separation. After thinking about the data more though I was thinking that in reality those two features might be strongly correlated with eachother so there weouldn't be a strong point for me to use both of them in my analysis.
# 
# To try to solve this I will plot a Pearson Correlation matrix and try to pick out correlation between different features so I don't use correlated features in my model.

# In[44]:


plt.figure(figsize=(18,18))
corr = xFeats.corr()
sns.heatmap(corr, annot=True, linewidths = .5, cmap=plt.cm.Blues, fmt = '.1f')
plt.show()


# In[45]:


#now to select the highly correlated features
cor_target = abs(corr['area_mean'])

relevant_features = cor_target[cor_target>0.5]
relevant_features


# It looks like area_mean was in fact correlated with perimeter_mean but also radius_mean. I wil go through a few more tests like this to find some more correlations. I'll pick what features to check by looking at the swarm plots.

# In[46]:


cor_target = abs(corr['concavity_mean'])
relevant_features = cor_target[cor_target>0.5]
relevant_features


# The concavity_mean feature is correlated with the compactness_mean feature and the concave points_mean feature.

# In[47]:


cor_target = abs(corr['radius_se'])
relevant_features = cor_target[cor_target>0.5]
relevant_features


# The radius_se feature is closely correlated with the perimeter_se feature and the area_se feature.

# In[48]:


cor_target = abs(corr['radius_worst'])
relevant_features = cor_target[cor_target>0.5]
relevant_features


# The radius_worst feature is closely correlated with the features: perimeter_mean, perimeter_worst, and radius_mean. Going through this process is giving me a list of features that I can drop from the classifier.

# In[49]:


cor_target = abs(corr['texture_mean'])
relevant_features = cor_target[cor_target>0.5]
relevant_features


# In[50]:


cor_target = abs(corr['area_worst'])
relevant_features = cor_target[cor_target>0.5]
relevant_features


# Now I will remove the features that aren't valuable to the model because of strong correlations to ther other features in the data.

# In[53]:


drops = ['perimeter_mean','radius_mean','compactness_mean', 'concave points_mean',
        'perimeter_se', 'area_se', 'perimeter_worst', 'texture_worst', 'area_worst',
        'compactness_se', 'concave points_se']
newXFeats = xFeats.drop(drops,axis=1)
newXFeats.head()


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(newXFeats, yDiag, test_size=0.2)

cl = RandomForestClassifier()
cl = cl.fit(X_train, y_train)
accur = accuracy_score(y_test,cl.predict(X_test))
print(accur)


# In[ ]:




