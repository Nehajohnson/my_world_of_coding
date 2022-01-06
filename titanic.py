#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # 1. Load the dataset into python environment

# In[57]:


data=pd.read_csv('titanic_dataset.csv')
data.head()


# # 2. Make ‘PassengerId’ as the index column

# In[58]:


data1=data.set_index('PassengerId')
data1


# # 3. Check the basic details of the dataset

# In[59]:


data.shape


# The data contain 891 rows and 12 columns.

# In[60]:


data.info()


# In[61]:


data1.describe()


# In[62]:


plt.figure(figsize=(10,8))
sns.heatmap(data1.corr(),cmap='BuPu',annot=True);


# In[63]:


There is no or very less correlation amoung the different columns in the dataset.


# In[64]:


data1.duplicated(keep='first').sum()


# There are no duplicated values in the dataset.

# # 4. Fill in all the missing values present in all the columns in the dataset

# In[65]:


data1.isnull().sum()


# There are total of 866 missing values in the dataset. With 177, 687 and 2 missing values in columns Age, Cabin and Embarked respectively.

# In[66]:


sns.heatmap(data1.isnull(),yticklabels=False,cbar=False)


# In[67]:


data1['Age']=data1['Age'].fillna(data1['Age'].mode()[0])
data1['Cabin']=data1['Cabin'].fillna(data1['Cabin'].mode()[0])
data1['Embarked']=data1['Embarked'].fillna(data1['Embarked'].mode()[0])


# We fill the missing values with the help of mode.

# In[68]:


data1.isnull().sum()


# # 5. Check and handle outliers in at least 3 columns in the dataset

# In[69]:


data1.hist(figsize=(10,10))
plt.show()


# From the above graphs we can see that there are outliers in the columns Sibsp, Parch and Fare

# Used the method of capping anf trimming to treat outliers.

# In[70]:


Q1 = data1['SibSp'].quantile(0.25)
Q3 = data1['SibSp'].quantile(0.75)
IQR = Q3 - Q1
whisker_width = 1.5
lower_whisker = Q1 -(whisker_width*IQR)
upper_whisker = Q3 + (whisker_width*IQR)
data1['SibSp']=np.where(data1['SibSp']>upper_whisker,upper_whisker,np.where(data1['SibSp']<lower_whisker,lower_whisker,data1['SibSp']))


# In[71]:


Q1 = data1['Parch'].quantile(0.25)
Q3 = data1['Parch'].quantile(0.75)
IQR = Q3 - Q1
whisker_width = 1.5
lower_whisker = Q1 -(whisker_width*IQR)
upper_whisker = Q3 + (whisker_width*IQR)
data1['Parch']=np.where(data1['Parch']>upper_whisker,upper_whisker,np.where(data1['Parch']<lower_whisker,lower_whisker,data1['Parch']))


# In[73]:


Q1 = data1['Fare'].quantile(0.25)
Q3 = data1['Fare'].quantile(0.75)
IQR = Q3 - Q1
whisker_width = 1.5
lower_whisker = Q1 -(whisker_width*IQR)
upper_whisker = Q3 + (whisker_width*IQR)
data1['Fare']=np.where(data1['Fare']>upper_whisker,upper_whisker,np.where(data1['Fare']<lower_whisker,lower_whisker,data1['Fare']))


# In[19]:


data2=data1[['SibSp','Parch','Fare']]
data2.hist(figsize=(10,10))
plt.show()


# In[54]:


data2.describe()


# We can see from above graphs that outliers are removed.

# # 6. Do min max scaling on the feature set (Take ‘Survived’ as target)

# In[38]:


from sklearn.preprocessing import MinMaxScaler
x=data1.drop(['Survived', 'Name', 'Ticket', 'Fare', 'Cabin' ],axis=1)
x


# In[23]:


x.info()


# In[43]:


x=pd.get_dummies(x)
x


# In[48]:


x.info()


# In[51]:


x.describe()


# In[52]:


scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.to_numpy())
x_scaled = pd.DataFrame(data1_scaled, columns=['Pclass','Age','SibSp','Parch','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S'])
print("Scaled Dataset Using MinMaxScaler")
x_scaled.head()


# In[53]:


x_scaled.describe()


# In[ ]:




