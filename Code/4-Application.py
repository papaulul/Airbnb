#%%
import os 
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
file_path = 'LA-2b.csv'
df_new_city = pd.read_csv(file_path,index_col = 0)
#%%
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from xgboost import XGBClassifier
# All important features will be put here 

#%%
files = os.listdir('models')
best = []
for filename in files: 
    model = pickle.load(open("models/"+filename, 'rb'))
    X = df_new_city.drop('isPlus', axis=1)
    y = df_new_city['isPlus']
    predictions = model.predict(X)
    TN,FP,FN,TP = confusion_matrix(y, predictions).ravel()
    met = (float(TP)+float(FP))/(float(TP) + float(FP)+ float(FN))
    if TN+ FN > 10000 and TP > 200:
        print(filename, met,"\nTN:",TN,'\nFN',FN,"\nTP:", TP,'\nFP',FP)
