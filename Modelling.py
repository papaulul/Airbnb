#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../2019 Spring/SpringAccel'))
	print(os.getcwd())
except:
	pass
sns.set_style('whitegrid')
sns.set_palette("husl")
pd.options.display.max_columns = None


#%%
df = pd.read_csv('Airbnb_Listings_with_Amenities.csv',index_col = 0)

#%%
df.head()

#%%
print("Has of NA df: {}".format(any(df.isna())))
#%%
df.describe()

#%%
for i in df.columns: 
    if any(df[i].isna()):
        print("{} has {} NA".format(i,sum(df[i].isna())))
        print("Plus listings: {}".format(sum(df[df[i].isna()]['isPlus'].dropna())))
#%%
df[(df['zipcode'].isna()) & (df['isPlus']==1)][['latitude','longitude', 'zipcode']]
# Using longitude and latitude, we were able to fill in the zipcode 
df.set_value(630, 'zipcode', 90404)
#%%
df.drop('host_neighbourhood',axis=1,inplace=True)

#%%
df.drop('amenities',axis=1,inplace=True)

#%%
df['host_response_time'].value_counts()

#%%
newvar = list(df[(df['host_response_time'].isna())].reset_index()['index'])
for i in newvar:
    df.set_value(i, 'host_response_time','Unavaliable')

#%%
for i in df.columns: 
    if any(df[i].isna()):
        print("{} has {} NA".format(i,sum(df[i].isna())))
        print("Plus listings: {}".format(sum(df[df[i].isna()]['isPlus'].dropna())))


#%%
# Dropping all rows with NA  cause doesn't affect Plus listings
df.dropna(axis=0,inplace = True)
for i in df.columns: 
    if any(df[i].isna()):
        print("{} has {} NA".format(i,sum(df[i].isna())))
        print("Plus listings: {}".format(sum(df[df[i].isna()]['isPlus'].dropna())))

#%%
df.info()


#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix


#%%
for i in df.columns: 
    print(i)

#%%
# Making Judgment call: Removing some columns might add back later

df_model = df.drop(['host_id','host_name','host_since','host_verifications','neighbourhood_cleansed','city','state','zipcode','market','smart_location','id','Unnamed: 61'], axis=1)

#%%
items=list(df_model.select_dtypes('object').columns)
for cols in items:
    add = pd.get_dummies(df_model[cols],drop_first=True)
    df_model=pd.concat([df_model,add], axis=1)

df_model.drop(items,axis=1, inplace= True)

#%%
X_train, X_test, y_train, y_test = train_test_split(df_model.drop('isPlus',axis= 1),df_model['isPlus'],train_size = 0.8181818182,stratify=df_model['isPlus'])

#%%
# Best Model So far
"""
The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as 
n_samples / (n_classes * np.bincount(y)).

"""
logmodel = LogisticRegression(class_weight='balanced' ,max_iter=50,n_jobs=-1)
logmodel.fit(X_train,y_train)
predictions_log = logmodel.predict(X_test)
print(classification_report(y_test, predictions_log))
print(confusion_matrix(y_test, predictions_log))

"""
Problem: Since we don't have the full list, it is possible that FP 
can be much more since we only have 240 out of the 900+ plus listings. 
Is this a big problem?  
"""

#%%

rfc = RandomForestClassifier(n_estimators=150,n_jobs=-1)
rfc.fit(X_train,y_train)
predictions_rfc = rfc.predict(X_test)
print(classification_report(y_test, predictions_rfc))
print(confusion_matrix(y_test, predictions_rfc))


#%%
model = SVC()
model.fit(X_train,y_train)
predictions_svc = model.predict(X_test)
print(classification_report(y_test, predictions_svc))
print(confusion_matrix(y_test, predictions_svc))


#%%
print("These are the scores\nLogistic: {}\nRandomForest: {}\nSVC: {}".format(
    logmodel.score(X_test,y_test),rfc.score(X_test,y_test),model.score(X_test,y_test)))


#%%
df_copy = df_model
important_cols = ['Bathroom essentials', 'Breakfast table', 'Ceiling fan', 'En suite bathroom', 'Espresso machine', 'Full kitchen', 'Gas oven', 'HBO GO', 'Hot water kettle', 'Memory foam mattress', 'Mini fridge', 'Outdoor seating', 'Pillow-top mattress', 'Smart TV', 'Sun loungers', 'Walk-in shower', 'Netflix']
for i in df_model.columns:
    try: 
        if df_copy[i].dtype == 'bool' and i not in important_cols:
            df_copy.drop(i,axis=1,inplace = True)
    except:
        pass
#%%
X_trainx, X_testx, y_trainx, y_testx = train_test_split(df_copy.drop('isPlus',axis= 1),df_copy['isPlus'],train_size = 0.8181818182,stratify=df_copy['isPlus'])

#%%
logmodelx = LogisticRegression(class_weight='balanced')
logmodelx.fit(X_trainx,y_trainx)
predictions_log = logmodelx.predict(X_testx)
print(classification_report(y_testx, predictions_log))
print(confusion_matrix(y_testx, predictions_log))

#%%
rfcx = RandomForestClassifier(n_estimators=150,n_jobs=-1)
rfcx.fit(X_trainx,y_trainx)
predictions_rfc = rfcx.predict(X_testx)
print(classification_report(y_testx, predictions_rfc))
print(confusion_matrix(y_testx, predictions_rfc))

#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%%
probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic for Random Forest')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%%
probs = logmodelx.predict_proba(X_testx)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_testx, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%%
probs = rfcx.predict_proba(X_testx)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_testx, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic for Random Forest')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%%
print("These are the scores\nLogistic: {}\nRandomForest: {}".format(
    logmodelx.score(X_testx,y_testx),rfcx.score(X_testx,y_testx)))

#%%
