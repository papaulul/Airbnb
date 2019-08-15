import os 
import pickle
import pandas as pd
import numpy as np
import math
def scaled(X):
    # Iniate Standard Scaler
    ss = pickle.load(open("models/scaler.pkl","rb"))
    # Making sure scaling floats
    to_scale = X.select_dtypes("float")
    # Getting transformed values
    scaled = ss.fit_transform(to_scale)
    # Setting values to table
    X[to_scale.columns] = pd.DataFrame(scaled, columns = to_scale.columns)
    return X
if __name__ == "__main__":
    os.chdir('/Users/pkim/Dropbox/Projects/SpringAccel')
    # Path to file
    file_path_la = 'files/july19/LA_2b.csv'
    file_path_sf = 'files/july19/SF_2b.csv'
    df_la = pd.read_csv(file_path_la,index_col = "Unnamed: 0")
    df_sf = pd.read_csv(file_path_sf,index_col = "Unnamed: 0")
    # Combining files
    df = pd.concat([df_la,df_sf]).reset_index().drop('index',1)
    # Creating copy so that have original values that aren't scaled
    df_copy = df.copy()
    # Set to be predicted on
    df_model = df.drop(['id','amenities','host_id'],1).copy()
    X = scaled(df_model)
    xgb = pickle.load(open("models/XGB_Final_Model.sav","rb"))
    # On the copy table, adding columns of probability and decisions
    df_copy['Plus_prob'] = xgb.predict_proba(X.drop("isPlus",1))[:,1]
    df_copy['Plus_pred'] = xgb.predict(X.drop("isPlus",1))
    # Shortening amenity list by only including those that I deemed most important
    amen = [x for x in df_copy.select_dtypes("bool").columns if "_" not in x and "isPlus" not in x]
    full_amen = []
    for ind, i in enumerate(df_copy[amen].values):
        small_amen = []
        for n,j in enumerate(list(i)):
            if j: 
                small_amen.append(amen[n])
        full_amen.append(small_amen)
    df_copy['amenities'] = pd.Series(full_amen)
    # Dropping indicator columns now
    df_copy = df_copy.drop(amen,1)
    # Getting non log versions 
    logs = [x for x in df_copy.select_dtypes("float").columns if "_log" in x]
    logs = list(map(lambda x: "_".join(x.split("_")[:-1]),logs))
    # Need to get latitude and longitudes
    df_la_latlong = 'files/july19/LA_2.csv'
    df_sf_latlong = 'files/july19/SF_2.csv'
    la = pd.read_csv(df_la_latlong,index_col = 0)
    sf = pd.read_csv(df_sf_latlong, index_col= 0)
    places = [la,sf]
    for place in places: 
        avg_longitude = place['longitude'].mean()
        avg_latitude = place['latitude'].mean()
        place['d_long'] = place['longitude'].apply(lambda x: abs(x - avg_longitude))
        place['d_lat'] = place['latitude'].apply(lambda x: abs(x-avg_latitude))
    df_latlong = pd.concat([la,sf]).reset_index().drop('index',1)
    df_latlong = df_latlong[logs+['id']] 
    logs = [x for x in df_copy.select_dtypes("float").columns if "_log" in x]
    # merging tables and dropping any columns with log transformations
    df_copy = df_copy.merge(df_latlong, how ="left",on ="id").drop(logs+['host_id'],1)
    # just will use total review score
    review_cols = [x for x in df_copy.columns if "review" in x]
    review_cols.remove("review_scores_rating")
    df_copy.drop(review_cols,axis=1,inplace=True)
    # Finding probabilty of Non Plus for rows that predicted Non Plus
    df_copy.loc[df_copy['Plus_pred']==False,'Plus_prob'] = 1 - df_copy[df_copy['Plus_pred']==False]['Plus_prob']
    # According to wikipedia, 1 degree in latitude is ~69 miles
    df_copy['d_lat'] = df_copy['d_lat']*69
    # According to google, 1 degree in longitude ~40 degrees is ~53 miles
    df_copy['d_long']=df_copy['d_long']*53
    # Using Euclidean distance 
    df_copy['dist'] = df_copy['d_lat']**2 + df_copy['d_long']**2
    df_copy['dist'] = df_copy['dist'].apply(math.sqrt)
    df_copy.drop(['d_lat','d_long'],1,inplace=True)
    # Demo file to use in website
    df_copy.to_csv("files/july19/demo.csv")