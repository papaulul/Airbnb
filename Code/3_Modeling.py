# In[1]:
## Imports
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time 
start_time = time.time()
os.chdir(os.path.join(os.getcwd(), '..'))
#%%
# Import from 2
file_path = 'files/july19/LA_2b.csv'
df = pd.read_csv(file_path,index_col = 0)

#%%
# Import modeling tools 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from xgboost import XGBClassifier
#%%
# Making Judgment call: Removing some columns might add back later
df_model = df.drop(['id','amenities','host_id'],1).copy()
#%%
# Contains all columns that make the cut
good_vars = ['isPlus']
# Training and testing on Bernoulli Naive Bayes, Logistic Regression with and without Feature Selection 
def success(Iter, model_Name, met, model ): 
    """
    Inputs: 
        Iter: Current iteration in loop, will be an int
        model_Name: String of the name of the model 
    What it does: 
        Saves the model's specification 
        creates new column in df_model with the current model's prediction 
    """
    filename = 'models/'+model_Name + str(Iter) + '.sav'
    pickle.dump(model, open(filename, 'wb'))
    colname = model_Name + str(Iter) + '_pred'
    df_model[colname] = model.predict(df_model.drop(good_vars,axis=1))
    good_vars.append(colname)
    print(X_train.shape)
    print(colname, " is saved with metric: ", met)
def met(TP,FP,FN): 
    """
    Inputs: 
        TP: True Positives
        FP: False Postitives
        FN: False Negatives 
    Objective: 
        Give an 'error' to the model based on FN 
        formula: 1 - (TP+FP)/(TP+FP +FN) = FN/(TP+FP+FN)
    """
    return (float(FN))/(float(TP) + float(FP)+ float(FN))
def model_train_predict_matrix(model_type): 
    """
    Input: 
        model_type: Looks for string, either BNB or log as of right now
    Output: 
        Trains model on train set
        makes prediction on X_test
        Will use that for the confusion matrix and will grab the True Positive and False Positive 
    """
    if model_type == 'bnb':
        model = bnb
    elif model_type == 'log':
        model = log
    elif model_type == 'rf':
        model = rf
    elif model_type == 'xgb':
        model = xgb
    else: 
        return [0,0]
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    _,FP,FN,TP = confusion_matrix(y_test, predictions).ravel()
    return [TP,FP,model ,FN]
#%%
good_vars = ['isPlus']

#%%
import time 
start_time = time.time()
log_params = {
            'penalty':['l1','l2']
            }
xgb_params = {
        "max_depth":[depth for depth in range(1,15,2)],
        "learning_rate": [lr/100 for lr in range(1,10,1)],
        "n_estimators": [est for est in range(1,2001,500)]
        }
rf_params = {
        "max_depth":[depth for depth in range(1,15,2)],
        "max_features":[features for features in range(5,55,5)],
        "n_estimators": [est for est in range(1,2001,500)]
        }
bnb_params = {
            'alpha': [alpha/10 for alpha in range(0,10,1)]
            }

def rand_grid(params, model):
    grid = RandomizedSearchCV(model,params,n_iter=20)
    grid.fit(X_train,y_train)
    new_model = model.set_params(**grid.best_params_)
    print("Optimal Model Found")
    output_file = "files/" + str(new_model).split("(")[0]+".sav"
    pickle.dump(new_model,open(output_file, "wb"))
    return new_model


X_train, X_test, y_train, y_test = train_test_split(df_model.drop(good_vars,axis= 1),df_model['isPlus'],train_size = 0.75,stratify=df_model['isPlus'], random_state=999)

#log = rand_grid(log_params,LogisticRegression(solver="liblinear"))
#bnb = rand_grid(bnb_params,BernoulliNB())
#rf = rand_grid(rf_params, RandomForestClassifier())
#xgb = rand_grid(xgb_params,XGBClassifier())

log = pickle.load(open("files/LogisticRegression.sav","rb"))
bnb = pickle.load(open("files/BernoulliNB.sav","rb"))
rf = pickle.load(open("files/RandomForestClassifier.sav","rb"))
xgb = pickle.load(open("files/XGBClassifier.sav","rb"))


print("Parameter Search time in Seconds: ",time.time()-start_time)
start_time = time.time()
# Picking Best Model Based on KFold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(5, shuffle= True,random_state = 999)
scores = pd.DataFrame(columns=['bnb_met','log_met','rf_met','xgb_met'])


Iter = 1
for train_index, test_index in skf.split(df_model.drop('isPlus',axis=1),df_model['isPlus']):
    X_train, X_test = df_model.loc[train_index].drop(good_vars, axis=1),df_model.loc[test_index].drop(good_vars, axis=1)
    y_train, y_test = df_model['isPlus'].loc[train_index], df_model['isPlus'].loc[test_index]
    for sets in [X_train,X_test,y_train,y_test]:
        sets.dropna(axis=0,inplace=True)
    for m_name in ['bnb','log','rf','xgb']: 
        TP,FP, model, FN = model_train_predict_matrix(m_name)
        metric = met(TP,FP,FN)
        col = m_name +"_met"
        scores.loc[int(Iter),col] = metric
        print(col, str(Iter), metric)
    Iter += 1
for i in list(scores.columns): 
    scores[i]=scores[i].astype('float')
print(scores.describe())
print("K-fold time in Seconds: ",time.time()-start_time)

# Looking into undersampling 
number_of_plus = sum(df_model['isPlus'])
non_plus_index = df_model[df_model['isPlus'] == False].index
random_index = np.random.choice(non_plus_index,number_of_plus,replace=False)
plus_index = df_model[df_model['isPlus']==True].index
under_sampled_indexes = np.concatenate([plus_index,random_index])
under_sample = df_model.loc[under_sampled_indexes]
X_under = under_sample.drop('isPlus',1)
y_under = under_sample['isPlus']
X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(
    X_under,y_under,test_size = 0.3, random_state = 0)

lr_under = LogisticRegression()
lr_under.fit(X_under_train,y_under_train)
y_under_pred = lr_under.predict(X_under_test)


from sklearn.metrics import recall_score, accuracy_score
print("Undersampling Results: Recall, Accuracy, and Confusion Matrix")
print(recall_score(y_under_test,y_under_pred))
print(accuracy_score(y_under_test,y_under_pred))
print(confusion_matrix(y_under_test,y_under_pred))

log.fit(X_train,y_train)
pred = log.predict(X_test)

from sklearn.metrics import recall_score, accuracy_score
print("Normal Results: Recall, Accuracy, and Confusion Matrix")
print(recall_score(y_test,pred))
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))


pred_all = lr_under.predict(X_test)
print("Undersampling on Normal Data Results: Recall, Accuracy, and Confusion Matrix")
print(recall_score(y_test,pred_all))
print(accuracy_score(y_test,pred_all))
print(confusion_matrix(y_test,pred_all))

start_time = time.time()
Iter = 1
for train_index, test_index in skf.split(X_under,y_under):
    X_train, X_test = X_under.iloc[train_index],X_under.iloc[test_index]
    y_train, y_test = y_under.iloc[train_index], y_under.iloc[test_index]
    for sets in [X_train,X_test,y_train,y_test]:
        sets.dropna(axis=0,inplace=True)
    for m_name in ['bnb','log','rf','xgb']: 
        TP,FP, model, FN = model_train_predict_matrix(m_name)
        metric = met(TP,FP,FN)
        col = m_name +"_met"
        scores.loc[int(Iter),col] = metric
        print(col, str(Iter), metric)
    Iter += 1
for i in list(scores.columns): 
    scores[i]=scores[i].astype('float')
print(scores.describe())
print("K-fold Undersampling time in Seconds: ",time.time()-start_time)
