#!/usr/bin/env python
# coding: utf-8

# Id number: 1 to 214
# RI: refractive index
# Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# Mg: Magnesium
# Al: Aluminum
# Si: Silicon
# K: Potassium
# Ca: Calcium
# Ba: Barium
# Fe: Iron
# Type of glass: (class attribute) -- 1 building_windows_float_processed -- 2 building_windows_non_float_processed -- 3 vehicle_windows_float_processed -- 4 vehicle_windows_non_float_processed (none in this database) -- 5 containers -- 6 tableware -- 7 headlamps

# In[3]:


pip install mlxtend


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import seaborn as sns
#from sklearn import datasets, neighbors
from sklearn.linear_model import LogisticRegression
#from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import cross_val_score # import all the functions reqd for cross validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
#for scaling the data
from sklearn.preprocessing import StandardScaler
#for distances
from sklearn.metrics import classification_report
from scipy.spatial import distance


# In[11]:


df = pd.read_csv('trainKNN.txt')
df.shape


# In[12]:


attributes = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type_Of_Glass']
df.columns = attributes
df.head()


# In[13]:


df=df.drop(['Id'], axis=1)
df.head()


# In[14]:


df.isnull().sum()


# In[17]:


df = df.drop_duplicates()
df.shape


# In[18]:


df.describe()


# In[19]:


for k, v in df.items():
  q1 = v.quantile(0.25)
  q3 = v.quantile(0.75)
  irq = q3 - q1
  v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
  perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
  print("Column %s outliers = %.2f%%" % (k, perc))


# In[20]:


plt.figure(figsize = (16, 12))
sns.heatmap(df.corr(), annot = True, fmt = '.2%')


# In[21]:


b = []
for i in df.keys():
  b.append(i)
print(b)


# In[22]:


b.remove('Type_Of_Glass')
print(b)


# In[24]:


X = df[b].values#array of features
y = df['Type_Of_Glass'].values


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# In[26]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[27]:


for i in [1,2,3,4,5,6,7,8,9,10,20,25,30,35,40,45,50]:
  knn = KNeighborsClassifier(i,metric=distance.sqeuclidean) #initialising the model
  knn.fit(x_train,y_train) # training the model
  print("K value  : " , i, " score : ", np.mean(cross_val_score(knn, x_train, y_train, cv=4)))


# In[28]:


knn = KNeighborsClassifier(n_neighbors=5,metric=distance.sqeuclidean) #it will initialise the model with @neighbours as k 
knn.fit(x_train, y_train) # train the model
print("Train Accuracy : ", knn.score(x_train,y_train)) # test the model and it computes the accuracy (train data accuracy)
print("Val Accuracy : ", np.mean(cross_val_score(knn, x_train, y_train, cv=4)))


# In[29]:


df1 = pd.read_csv('testKNN.txt')


# In[30]:


attributes = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type_Of_Glass']
df1.columns = attributes
df1.head()


# In[31]:


df2=df1.drop(['Id'], axis=1)
df2=df2.drop(['Type_Of_Glass'], axis=1)
df2.head()


# In[32]:


type(df2)


# In[33]:


x_test
type(x_test)


# In[34]:


df2


# In[35]:


df2 = df2.values
df2


# In[38]:


df2_test = scaler.transform(df2)
results = knn.predict(df2_test)
print(results)


# In[39]:


df1['Type_Of_Glass_pred'] = results
df1


# In[40]:


for i in [1,2,3,4,5,6,7,8,9,10,20,25,30,35,40,45,50]:
  knn = KNeighborsClassifier(i,metric=distance.cityblock) #initialising the model
  knn.fit(x_train,y_train) # training the model
  print("K value  : " , i, " score : ", np.mean(cross_val_score(knn, x_train, y_train, cv=4)))


# In[41]:


knn = KNeighborsClassifier(n_neighbors=10,metric=distance.cityblock) #it will initialise the model with @neighbours as k 
knn.fit(x_train, y_train) # train the model
print("Train Accuracy : ", knn.score(x_train,y_train)) # test the model and it computes the accuracy (train data accuracy)
print("Val Accuracy : ", np.mean(cross_val_score(knn, x_train, y_train, cv=4)))


# In[43]:


df1 = pd.read_csv('testKNN.txt')
attributes = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type_Of_Glass']
df1.columns = attributes
df1.head()


# In[44]:


df2=df1.drop(['Id'], axis=1)
df2=df2.drop(['Type_Of_Glass'], axis=1)
df2.head()


# In[45]:


type(df2)


# In[46]:


x_test
type(x_test)


# In[47]:


df2


# In[48]:


df2 = df2.values
df2


# In[50]:


df2_test = scaler.transform(df2)
results = knn.predict(df2_test)
print(results)


# In[51]:


df1['Type_Of_Glass_pred'] = results
df1


# In[52]:


for k, v in df.items():
  q1 = v.quantile(0.25)
  q3 = v.quantile(0.75)
  irq = q3 - q1
  v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
  perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
  print("Column %s outliers = %.2f%%" % (k, perc))


# In[53]:


def floorcapping(df):
  i = input()
  Q1 = df[i].quantile(0.25)
  Q3 = df[i].quantile(0.75)
  IQR = Q3 - Q1
  whisker_width = 1.5
  lower_whisker = Q1 -(whisker_width*IQR)
  upper_whisker = Q3 + (whisker_width*IQR)
  x = ((df[i] < Q1 - whisker_width*IQR) | (df[i] > Q3 + whisker_width*IQR))
  x = pd.DataFrame(x) # convert to data frame
  # df[x.isin([True])]
  substring = 'True'
  y= x[x.apply(lambda row: row.astype(str).str.contains(substring, case=False).any(), axis=1)]
  if True in y[i].tolist():
    df[i]=np.where(df[i]>upper_whisker,upper_whisker,np.where(df[i]<lower_whisker,lower_whisker,df[i])) 
  # substitute upper and lower whiskes to outliers
floorcapping(df)


# In[56]:


floorcapping(df)


# In[57]:


def outlierpresence(df):
  for i in df.keys():
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    x = (df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))
    # df[x.isin([True])]
    substring = 'True'
    y= x[x.apply(lambda row: row.astype(str).str.contains(substring, case=False).any(), axis=1)] #IT WILL GIVE ALL OUTLIERS IN THE DATAFRAME WITH ALL COLUMNS
    if True in y[i].tolist(): #HERE WE CHECK True is in the list of particular column
      print('Outliers', '\033[1m'+ 'present' +'\033[0m', 'in the data of','\033[1m' + i + '\033[0m')
      print('-------------------------------')
    else:
      print('Outliers', '\033[1m'+ ' not present in the data of' +'\033[0m', 'in','\033[1m' + i + '\033[0m') 
      print('-------------------------------') 
outlierpresence(df)


# In[58]:


x_train = df.drop(['Type_Of_Glass'], axis=1)
x_train = x_train.values
x_train


# In[59]:


y_train = df['Type_Of_Glass']
y_train = y_train.values
y_train


# In[60]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)


# Euclidean Distance

# In[61]:


for i in [1,2,3,4,5,6,7,8,9,10,20,25,30,35,40,45,50]:
  knn = KNeighborsClassifier(i,metric=distance.sqeuclidean) #initialising the model
  knn.fit(x_train,y_train) # training the model
  print("K value  : " , i, " score : ", np.mean(cross_val_score(knn, x_train, y_train, cv=6))) 


# In[62]:



knn = KNeighborsClassifier(n_neighbors=6,metric=distance.sqeuclidean) #it will initialise the model with @neighbours as k 
knn.fit(x_train, y_train) # train the model
print("Train Accuracy : ", knn.score(x_train,y_train)) # test the model and it computes the accuracy (train data accuracy)
print("Val Accuracy : ", np.mean(cross_val_score(knn, x_train, y_train, cv=6)))


# In[63]:


df1 = pd.read_csv('testKNN.txt')
attributes = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type_Of_Glass']
df1.columns = attributes
df1.head()


# In[64]:


df2=df1.drop(['Id'], axis=1)
df2=df2.drop(['Type_Of_Glass'], axis=1)
df2.head()


# In[65]:


df2 = df2.values
df2


# In[67]:


df2_test = scaler.transform(df2)
results = knn.predict(df2_test)
print(results)


# In[68]:


df1['Type_Of_Glass_pred'] = results
df1


# Manhatton

# In[69]:


for i in [1,2,3,4,5,6,7,8,9,10,20,25,30,35,40,45,50]:
  knn = KNeighborsClassifier(i,metric=distance.cityblock) #initialising the model
  knn.fit(x_train,y_train) # training the model
  print("K value  : " , i, " score : ", np.mean(cross_val_score(knn, x_train, y_train, cv=6)))


# In[70]:


knn = KNeighborsClassifier(n_neighbors=8,metric=distance.cityblock) #it will initialise the model with @neighbours as k 
knn.fit(x_train, y_train) # train the model
print("Train Accuracy : ", knn.score(x_train,y_train)) # test the model and it computes the accuracy (train data accuracy)
print("Val Accuracy : ", np.mean(cross_val_score(knn, x_train, y_train, cv=6)))


# In[72]:


df1 = pd.read_csv('testKNN.txt')
attributes = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type_Of_Glass']
df1.columns = attributes
df1.head()


# In[73]:


df2=df1.drop(['Id'], axis=1)
df2=df2.drop(['Type_Of_Glass'], axis=1)
df2.head()


# In[74]:


df2 = df2.values


# In[75]:


df2_test = scaler.transform(df2)
results = knn.predict(df2_test)
print(results)


# In[78]:


df1['Type_Of_Glass_pred'] = results
df1


# I experimented the given data in two ways,
# 
# Initialize and fitting k-NN model by splitting training data
# 
# By using Euclidean metric :- 68%
# By using Manhattan metric :- 69%
# Again initialize and fitting k-NN model by without splitting training data and Clean the outliers from the features
# 
# By using Euclidean metric :- 67%
# By using Manhattan metric :- 68%
# In all of the above models they did't predict the glasses in 3rd and 4th class.
