# In[1]:
## Imports
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
am_i_local = "no"
if am_i_local == "yes":
    try:
        os.chdir(os.path.join(os.getcwd(), '../2019 Spring/SpringAccel'))
        print(os.getcwd())
    except:
        pass
from IPython.display import display
pd.options.display.max_columns = None
sns.set_style('whitegrid')
sns.set_palette("husl")
#%%
# Import from 2
file_path = 'SF-2b.csv'
df = pd.read_csv(file_path,index_col = 0)

#%%
# Import modeling tools 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from xgboost import XGBClassifier
#%%
# Making Judgment call: Removing some columns might add back later
df_model = df.reset_index().drop('index',axis=1).copy()
#%%
# Run a Jaccard similarity on nonPlus vs isPlus population
#%%
# Contains all columns that make the cut
good_model = ['isPlus']
# Training and testing on Bernoulli Naive Bayes, Logistic Regression with and without Feature Selection 
def success(Iter, model_Name, met ): 
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
    df_model[colname] = model.predict(df_model.drop(good_model,axis=1))
    good_model.append(colname)
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
    return (float(TP))/(float(TP) + float(FP)+ float(FN))
def model_train_predict_matrix(model_type): 
    """
    Input: 
        model_type: Looks for string, either BNB or log as of right now
    Output: 
        Trains model on train set
        makes prediction on X_test
        Will use that for the confusion matrix and will grab the True Positive and False Positive 
    """
    if model_type == 'BNB':
        model = BernoulliNB()
    elif model_type == 'log':
        model = LogisticRegression(class_weight='balanced')
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators = 1500, max_features = 20, n_jobs = -1, warm_start=False)
    elif model_type == 'xgb':
        model = XGBClassifier(verbosity = 0)
    else: 
        return [0,0]
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    _,FP,FN,TP = confusion_matrix(y_test, predictions).ravel()
    return [TP,FP,model ,FN]
#%%
# K-Fold to determine the best model 
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(4, shuffle= True)

#%%
scores = pd.DataFrame(columns=['BNB_met','log_met','rf_met','xgb_met'])
good_model = ['isPlus']
Iter = 1
for train_index, test_index in skf.split(df_model.drop('isPlus',axis=1),df_model['isPlus']):
    X_train, X_test = df_model.loc[train_index].drop(good_model, axis=1),df_model.loc[test_index].drop(good_model, axis=1)
    y_train, y_test = df_model['isPlus'].loc[train_index], df_model['isPlus'].loc[test_index]
    # BNB
    for i in [X_train,X_test,y_train,y_test]:
        i.dropna(axis=0,inplace=True)
    
    for m_name in ['BNB','log','rf','xgb']: 
        TP,FP, model, FN = model_train_predict_matrix(m_name)
        metric = met(TP,FP,FN)
        col = m_name +"_met"
        scores.loc[int(Iter),col] = metric
    Iter += 1
for i in scores.columns: 
    scores[i]=scores[i].astype('float')
print(scores.describe())
#%%
scores = pd.DataFrame(columns=['BNB_met','log_met','rf_met','xgb_met'])
epochs = 20
for Iter in range(1,epochs+1):
    # New train test
    X_train, X_test, y_train, y_test = train_test_split(df_model.drop(good_model,axis= 1),df_model['isPlus'],train_size = 0.75,stratify=df_model['isPlus'])
    for m_name in ['BNB','log','rf','xgb']: 
    #for m_name in ['log']: 
        TP,FP, model, FN = model_train_predict_matrix(m_name)
        metric = met(TP,FP,FN)
        col = m_name +"_met"
        scores.loc[int(Iter),col] = metric     
        if metric > 0.8:
            success(Iter, m_name,metric)
        #success(Iter, m_name,metric)
#%% 
for i in scores.columns: 
    scores[i]=scores[i].astype('float')
#%% 
print(scores.describe()) 
#%%
for i in df_model.select_dtypes('uint8'):
    df_model[i] = df_model[i].astype('bool')
#%%
preds = list(df_model[df_model.columns[len(df.columns):]].columns)
#%%
from sklearn.metrics.pairwise import pairwise_distances
feats = []
for a in good_model:
    df_model[a] = df_model[a].astype('float')
for a in good_model: 
    df_model[a] = df_model[a].astype('bool')
    d_base_bool=df_model.select_dtypes(include='bool')
    ###Similarity
    jac_sim=1-pairwise_distances(d_base_bool.T, metric = "jaccard")
    jac_sim = pd.DataFrame(jac_sim, index=d_base_bool.columns, columns=d_base_bool.columns)
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
        if j not in new and j not in feats[0]:
            new.append(j)
print("Here are the new features we found from modeling : ", new)
#%%
"""
# Look at feature importance later
from sklearn.ensemble import RandomForestRegressor
X = df_features_only.drop('isPlus',axis=1)
Y = df_model[preds[0]]
names = X.columns
rf = RandomForestRegressor(criterion='mae')
rf.fit(X, Y)
print("Features sorted by their score:")
for i in (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True)):
    print(i)
"""
#%%
# Bootstrap 
"""
n = len(df_model)
choice = list(range(0,n))
ind = np.random.choice(choice, size = n, replace = True)
df_boot = df_model.loc[ind]
"""
#%%
