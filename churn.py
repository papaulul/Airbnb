#%% 
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from xgboost import XGBClassifier
#%%
df_model = pd.read_csv('Churn_Dataset.csv')
#%%

df_model.drop(['state','area code','phone number'], axis = 1 ,inplace =True) 
for i in ['international plan','voice mail plan']: 
    df_model[i] = df_model[i].apply(lambda x: 1 if x == 'yes' else 0)
#%%
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(4, shuffle= True)

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
    TN,FP,FN,TP = confusion_matrix(y_test, predictions).ravel()
    return [TN,TP,FP,model ,FN]

for train_index, test_index in skf.split(df_model.drop('churn',axis=1),df_model['churn']):
    X_train, X_test = df_model.loc[train_index].drop('churn', axis=1),df_model.loc[test_index].drop('churn', axis=1)
    y_train, y_test = df_model['churn'].loc[train_index], df_model['churn'].loc[test_index]
    # BNB
    for i in [X_train,X_test,y_train,y_test]:
        i.dropna(axis=0,inplace=True)
    
    for m_name in ['BNB','log','rf','xgb']: 
        TN,TP,FP, model, FN = model_train_predict_matrix(m_name)
        print(m_name, ": ", str((TN+TP)/(TN+TP+FP+FN)))
        print(m_name, ": \nTP = ", TP, ": \nFP = ", FP,": \nFN = ", FN,": \nTN = ", TN)


#%%
from sklearn.ensemble import RandomForestRegressor
X = df_model.drop('churn',axis=1)
Y = df_model['churn']
names = X.columns
rf = RandomForestRegressor(criterion='mae')
rf.fit(X, Y)
print("Features sorted by their score:")
for i in (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True)):
    print(i)
