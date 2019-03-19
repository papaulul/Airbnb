#!/usr/bin/env python
# coding: utf-8

# In[1]:
## Imports
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

# Import from 2
df = pd.read_csv('Airbnb_Listings_with_Amenitiespt2.csv',index_col = 0)


# In[3]:
# Removed variables with high correlations 
d_list = ['Bath towel','Bedroom comforts','Body soap','Dishes and silverware','Cooking basics','Dryer','Wide clearance to shower','amenities','availability_30','availability_365','availability_60','availability_90']
df.drop(d_list,inplace=True, axis=1)
df.head()

# In[11]:

# Finds index of where host_resposne_time is na and then set NAN value as unavaliable
newvar = list(df[(df['host_response_time'].isna())].reset_index()['index'])
for i in newvar:
    df.set_value(i, 'host_response_time','Unavaliable')


# In[12]:
## Drop Column
df.drop(['host_id','host_name','host_since','host_verifications','neighbourhood_cleansed','city','state','zipcode','market','smart_location','id','Unnamed: 61','translation missing: en.hosting_amenity_49','translation missing: en.hosting_amenity_50','Toilet paper','listing_count_LA','number_of_amenities','host_neighbourhood','weekly_price'],axis=1, inplace=True)
# In[13]:


# Dropping all rows with NA  cause doesn't affect Plus listings
df.dropna(axis=0,inplace = True)


# In[14]:




# In[15]:

# Import modeling tools 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing



# In[17]:


# Making Judgment call: Removing some columns might add back later
df_model = df.copy()
# In[18]:
# Changes an integer types into boolean 
i64 = list(df_model.select_dtypes('int64').columns)
for i in i64:
    df_model[i] = df_model[i].astype('bool')
# Changes object types into dummy variables 
items=list(df_model.select_dtypes('object').columns)
for cols in items:
    add = pd.get_dummies(df_model[cols],drop_first=True)
    df_model=pd.concat([df_model,add], axis=1)
# drop original categorical variables
df_model.drop(items,axis=1, inplace= True)

# In[19]:
# Run a Jaccard similarity on nonPlus vs isPlus population
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
# All important features will be put here 
feats = []
# confirms that isPlus is boolean
df_model['isPlus'] = df_model['isPlus'].astype('bool')
# grabs all boolean variables 
d_base_bool=df_model.select_dtypes(include='bool')

###Similarity
jac_sim=1-pairwise_distances(d_base_bool.T, metric = "jaccard")
# Dataframe, index and columns are the boolean variables 
jac_sim = pd.DataFrame(jac_sim, index=d_base_bool.columns, columns=d_base_bool.columns)
# Gets isPlus similarities with all other variables sorted 
isPlus_simil=jac_sim['isPlus'].sort_values(ascending=False)

# Starts frrom the beginning
c_important=[isPlus_simil.index[1]]
# Trying to find when we remove the most important feature, what would be the next important feature
for i in isPlus_simil.index[1:]:
    max_sim=0
    for j in c_important:
        if i!=j:
            max_sim=max(max_sim,jac_sim.loc[i,j])
    if max_sim>0 and max_sim<0.6 and jac_sim.loc[i,a]>.1:
        c_important.append(i)
# Append the important columns 
feats.append(c_important)
df_model['isPlus'] = df_model['isPlus'].astype('float')

#%%
important_cols = feats
#%%
# Setting up empty Dataframe that will contain all scores from modeling 
scores = pd.DataFrame(columns=['BNB_TP','BNB_FP','log_TP','log_FP','BNB_TP_f','BNB_FP_f','log_TP_f','log_FP_f'])
# Training and testing on Bernoulli Naive Bayes, Logistic Regression with and without Feature Selection 
for j in range(1,11):
    # New train test
    X_train, X_test, y_train, y_test = train_test_split(df_model.drop('isPlus',axis= 1),df_model['isPlus'],train_size = 0.75,stratify=df_model['isPlus'])
    # BNB
    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train,y_train)
    predictions_BNB = BNBmodel.predict(X_test)
    _,BNB_FP,_,BNB_TP = confusion_matrix(y_test, predictions_BNB).ravel()
    if int(BNB_TP) == 68:
        filename = 'BNB' + str(j) + '.sav'
        #pickle.dump(BNBmodel, open(filename, 'wb'))
        print('saved')
    # Log
    logmodel = LogisticRegression(class_weight='balanced')
    logmodel.fit(X_train,y_train)
    predictions_log = logmodel.predict(X_test)   
    _,log_FP,_,log_TP = confusion_matrix(y_test, predictions_log).ravel()
    if int(log_TP) == 68:
        filename = 'log' + str(j) + '.sav'
        #pickle.dump(logmodel, open(filename, 'wb'))
        print('saved')

    # feature selection    
    for k in [X_train, X_test]:
        for i in k.columns:
            try: 
                if k[i].dtype == 'bool' and i not in important_cols:
                    k.drop(i,axis=1,inplace = True)
            except:
                    pass
    # BNB w/ Feature selection
    BNBmodel_f = BernoulliNB()
    BNBmodel_f.fit(X_train,y_train)
    predictions_BNB_f = BNBmodel_f.predict(X_test)    
    _,BNB_FP_f,_,BNB_TP_f = confusion_matrix(y_test, predictions_BNB_f).ravel()
    if int(BNB_TP_f) == 68:
        filename = 'BNB_f' + str(j) + '.sav'
        #pickle.dump(BNBmodel_f, open(filename, 'wb'))
        print('saved')

    # Log w/ Feature selection
    logmodel_f = LogisticRegression(class_weight='balanced')
    logmodel_f.fit(X_train,y_train)
    predictions_log_f = logmodel_f.predict(X_test)
    _,log_FP_f,_,log_TP_f = confusion_matrix(y_test, predictions_log_f).ravel()
    if int(log_TP_f) == 68:
        filename = 'log_f' + str(j) + '.sav'
        #pickle.dump(logmodel_f, open(filename, 'wb'))
        print('saved')

    scores.loc[j] = [ BNB_TP , BNB_FP , log_TP , log_FP , BNB_TP_f , BNB_FP_f , log_TP_f , log_FP_f ]

#%% 
for i in scores.columns: 
    scores[i]=scores[i].astype('int')
#%% 
scores.describe()
#%%
scores['BNB_TP'] + scores['BNB_FP']
scores['BNB_TP_f'] + scores['BNB_FP_f']
scores['log_TP'] + scores['log_FP']
scores['log_TP_f'] + scores['log_FP_f']

#%%
good_model = ['isPlus']
for j in range(1,11):
    # New train test
    df_copy = df_model.drop(good_model,axis=1).copy()

    X_train, X_test, y_train, y_test = train_test_split(df_copy,df_model['isPlus'],train_size = 0.75,stratify=df_model['isPlus'])
    # BNB
    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train,y_train)
    predictions_BNB = BNBmodel.predict(X_test)
    _,BNB_FP,_,BNB_TP = confusion_matrix(y_test, predictions_BNB).ravel()

    if int(BNB_TP)  >= 66:
        colname = 'BNB' + str(j) + '_pred'
        df_model[colname] = BNBmodel.predict(df_model.drop(good_model,axis=1))
        good_model.append(colname)
        print("done")
    # Log
    logmodel = LogisticRegression(class_weight='balanced')
    logmodel.fit(X_train,y_train)
    predictions_log = logmodel.predict(X_test)   
    _,log_FP,_,log_TP = confusion_matrix(y_test, predictions_log).ravel()

    if int(log_TP)  >= 66:
        colname = 'log' + str(j) + '_pred'
        df_model[colname] = logmodel.predict(df_model.drop(good_model,axis=1))
        good_model.append(colname)
        print("done")
    

    for k in [df_copy]:
        for i in k.columns:
            try: 
                if k[i].dtype == 'bool' and i not in important_cols:
                    k.drop(i,axis=1,inplace = True)
            except:
                    pass

    # feature selection    
    for k in [X_train, X_test]:
        for i in k.columns:
            try: 
                if k[i].dtype == 'bool' and i not in important_cols:
                    k.drop(i,axis=1,inplace = True)
            except:
                    pass
    # BNB w/ Feature selection
    BNBmodel_f = BernoulliNB()
    BNBmodel_f.fit(X_train,y_train)
    predictions_BNB_f = BNBmodel_f.predict(X_test) 
    _,BNB_FP_f,_,BNB_TP_f = confusion_matrix(y_test, predictions_BNB_f).ravel()
   
    if int(BNB_TP_f)  >= 66:
        colname = 'BNB_f' + str(j) + '_pred'
        df_model[colname] = BNBmodel_f.predict(df_copy)
        good_model.append(colname)
        print("done")


    # Log w/ Feature selection
    logmodel_f = LogisticRegression(class_weight='balanced')
    logmodel_f.fit(X_train,y_train)
    predictions_log_f = logmodel_f.predict(X_test)
    _,log_FP_f,_,log_TP_f = confusion_matrix(y_test, predictions_log_f).ravel()

    if int(log_TP_f)  >= 66:
        colname = 'log_f' + str(j) + '_pred'
        df_model[colname] = logmodel_f.predict(df_copy)
        good_model.append(colname)
        print("done")


#%%
for i in df_model.select_dtypes('uint8'):
    df_model[i] = df_model[i].astype('bool')
#%%
preds = list(df_model[df_model.columns[-12:]].columns)

#%% 

#%%
def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df
df_model=df_column_uniquify(df_model)


#%%
from sklearn.metrics.pairwise import pairwise_distances
feats = []
for a in preds: 
    df_model[a] = df_model[a].astype('bool')
    d_base_bool=df_model.select_dtypes(include='bool')

    ###Similarity
    jac_sim=1-pairwise_distances(d_base_bool.T, metric = "jaccard")
    jac_sim = pd.DataFrame(jac_sim, index=d_base_bool.columns, columns=d_base_bool.columns)
    #isPlus_simil=jac_sim['isPlus'].drop_duplicates(keep='first').sort_values(ascending=False)
    isPlus_simil=jac_sim[a].sort_values(ascending=False)

    c_important=[isPlus_simil.index[1]]
    for i in isPlus_simil.index[1:]:
        max_sim=0
        for j in c_important:
            if i!=j:
                max_sim=max(max_sim,jac_sim.loc[i,j])
        if max_sim>0 and max_sim<0.6 and jac_sim.loc[i,a]>.1:
            c_important.append(i)
    feats.append(c_important)
    df_model[a] = df_model[a].astype('float')



#%%
new = []
for i in feats:
    for j in i:
        if j not in new:
            new.append(j)
print(new)


#%%
from sklearn.ensemble import RandomForestRegressor
X = df_copy.drop(preds,axis=1)
Y = df_model[preds[9]]
names = X.columns
rf = RandomForestRegressor(criterion='mae')
rf.fit(X, Y)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True))

#%%
for i in (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True)):
    print(i)



#%%
