#!/usr/bin/env python
# coding: utf-8

# # Employee Attrition Rate using Regression
# 
# ## Introduction
# 
# Artificial intelligence is commonly used in various trade circles to automate processes, gather insights on business, and speed up processes. You will use Python to study the usage of artificial intelligence in real-life scenarios - how AI actually impacts industries. 
# 
# Employees are the most important entities in an organization. Successful employees offer a lot to organizations. In this notebook, we will use AI to predict the attrition rate of employees or how often a company can retain employees.
# 
# ## Context
# 
# I will be working with the dataset containing employee attrition rates, which is collected by Hackerearth and uploaded at [Kaggle](https://www.kaggle.com/blurredmachine/hackerearth-employee-attrition). I will use regression to predict attrition rates and see how successful is our model.
# 
# 
# 
# ## Use Python to open CSV files
# 
# We will the [scikit-learn](https://scikit-learn.org/stable/) and [pandas](https://pandas.pydata.org/) to work with our dataset. Scikit-learn is a very useful machine learning library that provides efficient tools for predictive data analysis.  Pandas is a popular Python library for data science. It offers powerful and flexible data structures to make data manipulation and analysis easier.
# 
# 
# ## Import Libraries
# 

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error


# ### Importing the Dataset
# 
# The dataset contains employee attrition rates. Let us visualize the dataset.
# 
# 

# In[3]:


train = pd.read_csv("[Dataset]_Train_(Employee).csv") 
test = pd.read_csv("[Dataset]_Test_(Employee).csv")


# ## Printing the columns of the training set

# In[4]:


#code for displaying names of columns
train=train.columns
print('names of columns in the train dataset is', train)


# In[5]:


print(train.shape)


# In[8]:


train.head()


# ### Data Description
# 
# Let us see how the data is distributed. We can visualize the mean, max, and min value of each column alongside other characteristics.

# In[7]:


train.describe()


# ## Task2: Get information about the training data set using the describe function

# In[18]:


#code for the describe function 
train.describe().T


# In[19]:


# Let's see if training set has any missing values
train.isna().any()


# ### Data Visualization
# 
# Now, let us see the correlation matrix to see how related are the features.

# In[9]:


plt.figure(figsize=(18,10))
cor = train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Accent)
plt.show()
plt.savefig("main_correlation.png")


# ### Preparing the model
# 
# Finalizing the data for the training and prepare the model.

# In[21]:


#Attrition_rate is the label or output to be predicted
#features will be used to predict Attrition_rate
label = ["Attrition_rate"]
features = ['VAR7','VAR6','VAR5','VAR1','VAR3','growth_rate','Time_of_service','Time_since_promotion','Travel_Rate','Post_Level','Education_Level']


# In[22]:


featured_data = train.loc[:,features+label]
#We will drop the columns here which have missing values using dropna function
featured_data = featured_data.dropna(axis=0)
featured_data.shape


# In[ ]:


X = featured_data.loc[:,features]
y = featured_data.loc[:,label]


# In[ ]:


#Here the training and test data are split 55% to 45% as test size is 0.55
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.55)


# In[ ]:


df = Linearregression()
df.fit(X_train,y_train)
y_pred = df.predict(X_test)
c = []
for i in range 


# In[ ]:


#Let's print the accuracy now
score = 100* max(0, 1-mean_squared_error(y_test, y_pred))
print(score)


# In[ ]:


df.evaluate()


# In[ ]:


#Predicting
import pandas as pd
dff = pd.DataFrame({'Employee_ID':test['Employee_ID'],'Attrition_rate':pf})
dff.head(10)


# ## Task 3: Print the first 20 columns of predictions
# 

# In[ ]:


dff.head(20)


# ### Conclusion
# 
# In this notebook, we have seen how AI can be used by companies to predict which employess would be loyal to them. We have bulit a linear regression model to predict the attrition rate.

# In[ ]:




