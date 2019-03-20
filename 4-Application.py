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
#%% 
file_path = 'SanFran-2.csv'
df_new_city = pd.read_csv(file_path,index_col = 0)

#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
# All important features will be put here 

def jaccard(df_model):
    feats = []
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
        if max_sim>0 and max_sim<0.6 and jac_sim.loc[i,'isPlus']>.1:
            c_important.append(i)
    # Append the important columns 
    feats.append(c_important)
    df_model['isPlus'] = df_model['isPlus'].astype('float')

    #%%
    important_cols = feats[0]
    print("Here are the important columns: ", important_cols)
jaccard(df_new_city)

#%%
filename = 'log1.sav'
model =   pickle.load(open(filename, 'rb'))

