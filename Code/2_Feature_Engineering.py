# Imports
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import display
from sklearn.linear_model import LinearRegression
import pickle
import time
import os
import warnings 
start_time = time.time()
warnings.filterwarnings('ignore')


def feature_engineering(city,list_of_amenities):
## Import csv from 1b 
    input_path= 'files/july19/'+city+'_1b.csv'
    df = pd.read_csv(input_path, index_col='Unnamed: 0')
    if city == "SF":
        df.drop("license",1,inplace=True)
    # Calculations number of Amenities avaliable 
    df['number_of_amenities'] = df['amenities'].apply(lambda x: len(x.split(','))) 
    # Calculates unique number of Amenities avaliable 
    df['number_of_unique_amenities'] = df['amenities'].apply(lambda x: len(set(x.split(',')))) 
    # Calculates number of Verifications host provides 
    df['number_of_host_verifications'] = df['host_verifications'].apply(lambda x: len(set(x.split(','))))
    df['amenities'] = df['amenities'].apply(lambda x: sorted([col.replace('"','').strip() for col in x[1:-1].split(",") if col != ""])) 
    versus = df[['id','amenities']]
    for i in list_of_amenities:
        versus[i] = False
    for i,j in enumerate(versus['amenities']):
        for k in j:
            if '"' in k:
                k = k.replace('"',"")
            versus.set_value(i,k, True) 
    versus.drop('amenities', axis=1, inplace = True)
    versus.dropna(axis=0,inplace=True)
    df = df.merge(versus, on = 'id', how = 'left')
    df = remove_corr(df,city)
    ## Drop Column with high sparsity or other issues 
    df.drop(['host_verifications','translation missing: en.hosting_amenity_49','translation missing: en.hosting_amenity_50'],axis=1, inplace=True)
    #Verify that none of the rows with Plus listings have null 
    # Dropping all rows with NA  cause doesn't affect Plus listings
    # Changes object types into dummy variables 
    items=[categorical for categorical in list(df.select_dtypes('object').columns) if categorical not in list_of_amenities and categorical != "amenities" and categorical != "id"]
    for cols in items:
        add = pd.get_dummies(df[cols],drop_first=True,prefix=cols).astype('bool')
        df=pd.concat([df,add], axis=1)
    # Changes an integer types into boolean 
    # drop original categorical variables
    df.drop(items,axis=1, inplace= True)
    df['isPlus'] = df['isPlus'].apply(lambda x: 1 if x == 1 else 0).astype('bool')   
    return df

def missing_inpute(df, city):
    output_path = 'files/july19/'+city+'_2.csv'
    bedroom_1 = df[df['bathrooms']==1]['bedrooms'].dropna().mean()
    bedroom_1_ind = df[(df['bathrooms']==1) & (df['bedrooms'].isna())].index
    bedroom_2 = df[df['bathrooms']==2]['bedrooms'].dropna().mean()
    bedroom_2_ind = df[(df['bathrooms']==2) & (df['bedrooms'].isna())].index
    bathroom_1 = df[df['bedrooms']==1]['bathrooms'].dropna().mean()
    bathroom_1_ind = df[(df['bedrooms']==1) & (df['bathrooms'].isna())].index
    
    for ind in bedroom_1_ind:
        df.at[ind,'bedrooms'] = bedroom_1 
    
    for ind in bedroom_2_ind:
        df.at[ind,'bedrooms'] = bedroom_2 
        
    for ind in bathroom_1_ind: 
        df.at[ind,'bathrooms'] = bathroom_1 
    df_no_na = df.copy().dropna(0)
    if city =="LA":
        review_na_cols = [cols for cols in df_no_na.columns if "review_scores" in cols]
        object_cols = list(df_no_na.select_dtypes("object").columns)
        print(object_cols)    
        print(review_na_cols)
        print(len(object_cols+review_na_cols))
        X = df_no_na.drop(review_na_cols+object_cols, 1).as_matrix()
        for cols in review_na_cols:
            y = df_no_na[cols].dropna().as_matrix()
            to_pred = df[df[cols].isna()].drop(review_na_cols+object_cols,1).as_matrix()
            lin = LinearRegression()
            lin.fit(X,y)
            pred = lin.predict(to_pred)
            df[cols].loc[df[cols].isna()] = pred
            pickle.dump(lin, open("files/transformations/linear_missing_"+cols+".sav", 'wb'))
    else: 
        review_na_cols = [cols for cols in df_no_na.columns if "review_scores" in cols]
        object_cols = list(df_no_na.select_dtypes("object").columns + df_no_na.select_dtypes("bool").columns)
        print(object_cols)    
        print(review_na_cols)
        print(len(object_cols+review_na_cols))
        for cols in review_na_cols:
            to_pred = df[df[cols].isna()].drop(review_na_cols+object_cols,1).as_matrix()
            lin = pickle.load(open("files/transformations/linear_missing_"+cols+".sav",'rb'))
            pred = lin.predict(to_pred)
            df[cols].loc[df[cols].isna()] = pred

    for cols in review_na_cols:
        df[cols] = df[cols].apply(lambda x: 100 if x > 100 else int(x))
    #%%
    df.to_csv(output_path)

def merge_columns(LA,SF):
    la = list(LA.columns)
    sf = list(SF.columns)
    la_col = [col for col in la if col not in sf]
    sf_col = [col for col in sf if col not in la]
    return LA,SF

def amenities(LA,SF):
        #Expanding amenities to individual columns 
    list_of_all_amenities = []
    df_la = pd.read_csv(LA, index_col='Unnamed: 0')['amenities'].apply(lambda x: x[1:-1].replace('"',"").split(","))
    df_sf = pd.read_csv(SF, index_col='Unnamed: 0')['amenities'].apply(lambda x: x[1:-1].replace('"',"").split(","))
    df = pd.concat([df_la,df_sf])
    for i in df.values:
        for j in i: 
            j = j.strip()
            if j != "":
                list_of_all_amenities.append(j)
    list_of_all_amenities = sorted(list(set(list_of_all_amenities)))
    return list_of_all_amenities 


def remove_corr(df, city):
# Finding Variables that have high correlation 
    if city == "LA":
        high_corr = pd.DataFrame(df.corr().abs().unstack()[df.corr().abs().unstack().sort_values(kind="quicksort")>.8]).reset_index()
        d_list = list(high_corr[high_corr['level_0']!=high_corr['level_1']]['level_0'].unique())
        pickle.dump(d_list, open("files/july19/correlated_var.txt",'wb'))
    else: 
        d_list = pickle.load(open("files/july19/correlated_var.txt","rb"))
    if "isPlus" in d_list:
        d_list.remove("isPlus")
    df = df.drop(d_list, axis=1)
    return df



if __name__ == "__main__":
    os.chdir('/Users/pkim/Dropbox/Projects/SpringAccel')
    input_path_LA= 'files/july19/LA_1b.csv'
    input_path_SF= 'files/july19/SF_1b.csv'
    list_of_amenities = amenities(input_path_LA,input_path_SF)
    df_LA = feature_engineering('LA',list_of_amenities)
    df_SF = feature_engineering('SF',list_of_amenities)
    df_LA, df_SF = merge_columns(df_LA,df_SF)
    print("--- %s seconds ---" % (time.time() - start_time))
