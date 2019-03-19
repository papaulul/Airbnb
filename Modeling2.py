#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
try:
	os.chdir(os.path.join(os.getcwd(), '../2019 Spring/SpringAccel'))
	print(os.getcwd())
except:
	pass
from IPython.display import display

pd.options.display.max_columns = None


sns.set_style('whitegrid')
sns.set_palette("husl")


# In[2]:


df = pd.read_csv('Airbnb_Listings_with_Amenitiespt2.csv',index_col = 0)


# In[3]:


df.head()


# In[4]:


print("Has of NA df: {}".format(any(df.isna())))


# In[5]:


df.describe()


# In[6]:


for i in df.columns: 
    if any(df[i].isna()):
        print("{} has {} NA".format(i,sum(df[i].isna())))
        print("Plus listings: {}".format(sum(df[df[i].isna()]['isPlus'].dropna())))


# In[7]:


df[(df['zipcode'].isna()) & (df['isPlus']==1)][['latitude','longitude', 'zipcode']]
# Using longitude and latitude, we were able to fill in the zipcode 
df.set_value(630, 'zipcode', 90404)
df.drop('host_neighbourhood',axis=1,inplace=True)
df.drop('amenities',axis=1,inplace=True)


# In[10]:


df['host_response_time'].value_counts()


# In[11]:


newvar = list(df[(df['host_response_time'].isna())].reset_index()['index'])
for i in newvar:
    df.set_value(i, 'host_response_time','Unavaliable')


# In[12]:


for i in df.columns: 
    if any(df[i].isna()):
        print("{} has {} NA".format(i,sum(df[i].isna())))
        print("Plus listings: {}".format(sum(df[df[i].isna()]['isPlus'].dropna())))


# In[13]:


# Dropping all rows with NA  cause doesn't affect Plus listings
df.dropna(axis=0,inplace = True)
for i in df.columns: 
    if any(df[i].isna()):
        print("{} has {} NA".format(i,sum(df[i].isna())))
        print("Plus listings: {}".format(sum(df[df[i].isna()]['isPlus'].dropna())))


# In[14]:


df.info()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


# In[16]:


for i in df.columns: 
	print(i)


# In[17]:


# Making Judgment call: Removing some columns might add back later

df_model = df.drop(['host_id','host_name','host_since','host_verifications','neighbourhood_cleansed','city','state','zipcode','market','smart_location','id','Unnamed: 61'], axis=1)

# In[18]:


items=list(df_model.select_dtypes('object').columns)
for cols in items:
    add = pd.get_dummies(df_model[cols],drop_first=True)
    df_model=pd.concat([df_model,add], axis=1)

df_model.drop(items,axis=1, inplace= True)
#df_model = df_model.select_dtypes(include = ['float','int'])
#X_train, X_test, y_train, y_test = train_test_split(df_model.drop('isPlus',axis= 1),df_model['isPlus'],train_size = 0.8181818182,stratify=df_model['isPlus'])
X_train, X_test, y_train, y_test = train_test_split(df_model.drop('isPlus',axis= 1),df_model['isPlus'],train_size = 0.75,stratify=df_model['isPlus'])


# In[19]:
from sklearn.metrics import classification_report,confusion_matrix

NBmodel = GaussianNB()
NBmodel.fit(X_train,y_train)
predictions_NB = NBmodel.predict(X_test)
print(classification_report(y_test, predictions_NB))
print(confusion_matrix(y_test, predictions_NB))

#filename = 'NBmodel_finalized_model.sav'
#pickle.dump(NBmodel, open(filename, 'wb'))

#NBmodel = pickle.load(open(filename, 'rb'))

# In[20]:

BNBmodel = BernoulliNB()
BNBmodel.fit(X_train,y_train)
predictions_BNB = BNBmodel.predict(X_test)
print(classification_report(y_test, predictions_BNB))
print(confusion_matrix(y_test, predictions_BNB))

#filename = 'BNBmodel_finalized_model.sav'
#pickle.dump(BNBmodel, open(filename, 'wb'))

#BNBmodel = pickle.load(open(filename, 'rb'))

# In[]:

logmodel = LogisticRegression(class_weight='balanced')
logmodel.fit(X_train,y_train)
predictions_log = logmodel.predict(X_test)
print(classification_report(y_test, predictions_log))
print(confusion_matrix(y_test, predictions_log))

#filename = 'logmodel_finalized_model.sav'
#pickle.dump(logmodel, open(filename, 'wb'))

#logmodel = pickle.load(open(filename, 'rb'))

#%% 
df_copy = df_model.copy()
df_model['NB_pred'] = NBmodel.predict(df_model.drop('isPlus',axis=1))
df_model['BNBmodel_pred'] = BNBmodel.predict(df_model.drop(['isPlus','NB_pred'],axis=1))
df_model['logmodel_pred'] = logmodel.predict(df_model.drop(['isPlus','NB_pred','BNBmodel_pred'],axis=1))



# In[24]:
important_cols = ['Bathroom essentials', 'Breakfast table', 'Ceiling fan', 'En suite bathroom', 'Espresso machine', 'Full kitchen', 'Gas oven', 'HBO GO', 'Hot water kettle', 'Memory foam mattress', 'Mini fridge', 'Outdoor seating', 'Pillow-top mattress', 'Smart TV', 'Sun loungers', 'Walk-in shower', 'Netflix']
for j in [X_train, X_test]:
#for j in [df_copy]:
	for i in j.columns:
		try: 
			if j[i].dtype == 'bool' and i not in important_cols:
				j.drop(i,axis=1,inplace = True)
		except:
				pass

# In[21]:

LDAmodel = LinearDiscriminantAnalysis()
LDAmodel.fit(X_train,y_train)
predictions_LDA = LDAmodel.predict(X_test)
print(classification_report(y_test, predictions_LDA))
print(confusion_matrix(y_test, predictions_LDA))

#filename = 'LDAmodel_finalized_model.sav'
#pickle.dump(LDAmodel, open(filename, 'wb'))

#LDAmodel = pickle.load(open(filename, 'rb'))

# In[22]:

QDAmodel = QuadraticDiscriminantAnalysis()
QDAmodel.fit(X_train,y_train)
predictions_QDA = QDAmodel.predict(X_test)
print(classification_report(y_test, predictions_QDA))
print(confusion_matrix(y_test, predictions_QDA))

#filename = 'QDAmodel_finalized_model.sav'
#pickle.dump(QDAmodel, open(filename, 'wb'))

#QDAmodel = pickle.load(open(filename, 'rb'))

# In[23]:
df_copy['LDA_pred'] = LDAmodel.predict(df_copy.drop('isPlus',axis=1))
df_copy['QDA_pred'] = QDAmodel.predict(df_copy.drop(['isPlus','LDA_pred'],axis=1))

#%%
#df_model.drop(['BNBmodel_pred','logmodel_pred','isPlus'], axis=1).to_csv('NB_rows.csv')
#df_model.drop(['NB_pred','logmodel_pred','isPlus'], axis=1).to_csv('BNB_rows.csv')
#df_model.drop(['BNBmodel_pred','NB_pred','isPlus'], axis=1).to_csv('log_rows.csv')
##%%
#df_copy.drop(['QDA_pred','isPlus'],axis=1).to_csv('LDA_rows.csv')
#df_copy.drop(['LDA_pred','isPlus'],axis=1).to_csv('QDA_rows.csv')

#%%
import pandas_profiling as pp 

#%%
pp.ProfileReport(df_model)

#%%
