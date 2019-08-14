#%%
import os 
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import math
from sklearn.metrics import classification_report,confusion_matrix
pd.options.display.max_columns = None
sns.set_style('whitegrid')
sns.set_palette("husl")

if __name__ == "__main__":
    os.chdir('/Users/pkim/Dropbox/Projects/SpringAccel')
    # Reading in file
    file_path = 'files/july19/SF_2b.csv'
    # Same read in as LA
    df_new_city = pd.read_csv(file_path,index_col = "Unnamed: 0")
    df_new_city.drop(['id','amenities','host_id'],1,inplace=True)
    # Loading final model from 3b and predicting
    xgb = pickle.load(open("models/XGB_Final_Model.sav","rb"))
    predict = xgb.predict(df_new_city.drop("isPlus",1))
    # Evaluation
    print("SF Evaluation")
    print(classification_report(df_new_city['isPlus'],predict))
    print(confusion_matrix(df_new_city['isPlus'],predict))
