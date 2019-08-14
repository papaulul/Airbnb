
import os 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
import time
start_time = time.time()

def jaccard(df_model):
    feats = []
    # grabs all boolean variables 
    d_base_bool=df_model.select_dtypes(include='bool')
    d_base_bool = d_base_bool[[col for col in d_base_bool.columns if "_" not in col]]
    ###Similarity
    jac_sim=1-pairwise_distances(d_base_bool.T.as_matrix(), metric = "jaccard")
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
        if max_sim>0 and max_sim<0.6 and jac_sim.loc[i,'isPlus']>.07:
            c_important.append(i)
    # Append the important columns 
    feats.append(c_important)

    
    important_cols = feats[0]
    print("Here are the important columns: ", important_cols)
    return important_cols

def data_fix(file_path, file_path_sf, output_la, output_sf):
    la = pd.read_csv(file_path,index_col = 0)
    sf = pd.read_csv(file_path_sf, index_col= 0)
    # Now that they have the same columns, going to run a feature importance on the all boolean columns
    full = pd.concat([la,sf])

    important_cols = jaccard(full)

    # Going to drop any amenities that are not important 
    places = [la,sf]
    for place in places:
        for i in [cols for cols in place.columns if "_" not in place.columns]:
            try: 
                if place[i].dtype == 'bool' and i not in important_cols and i != 'isPlus':
                    place.drop(i,axis=1,inplace = True)
            except:
                    pass
        # Also dropping couple of others that were not important
        drop_list = ['host_has_profile_pic','host_identity_verified','host_is_superhost','instant_bookable',
                    'require_guest_phone_verification','require_guest_profile_picture','is_location_exact']
        place.drop(drop_list, axis=1, inplace=True)


    ## Latitude/longitude idea: Instead of lat/long, make it to distance from average
    for place in places: 
        avg_longitude = place['longitude'].mean()
        avg_latitude = place['latitude'].mean()
        place['d_long'] = place['longitude'].apply(lambda x: abs(x - avg_longitude))
        place['d_lat'] = place['latitude'].apply(lambda x: abs(x-avg_latitude))
        place.drop(['latitude','longitude'],axis=1,inplace=True)
        # Was lazy, so I decided here was the place to change interger columns to float
        place[place.select_dtypes('int').columns] = place.select_dtypes('int').astype('float')


    # Look at all float columns
    count = 1
    for ind,place in enumerate(places):
        for col in sorted(place.select_dtypes('float').columns): 
            plt.figure(count)
            plt.hist(col, data = place, range= tuple(place[col].quantile([.01,.99]).values))
            if ind == 0: 
                city = 'LA'
            else:
                city = 'SF'
            title = city+': '+col
            plt.title(title)
            plt.savefig("files/plots/"+title+".png")
            count+=1
    # Pulled any columns that were right skewed 
    # Except those that involve price. I didn't want to touch those columns
    right_skewed = ['bathrooms','bedrooms','calculated_host_listings_count_private_rooms',
                    'calculated_host_listings_count_shared_rooms','calendar_updated', 'cleaning_fee',
                    'd_lat','d_long','extra_people','guests_included','number_of_reviews','price','security_deposit']
    import math
    for place in places: 
        for skew in right_skewed:
            # trying to fix right_skewed with either sqrt or log
            #place[skew+"sqrt"] = place[skew].apply(lambda x: math.sqrt(abs(x)))
            place[skew+"log"] = place[skew].apply(lambda x: math.log(abs(x)) if abs(x) > 0 else 0)
        place.drop(right_skewed,axis=1,inplace=True)
    # checking to see if we fixed the skewed data
    count = 1
    for ind,place in enumerate(places):
        for col in sorted(place.select_dtypes('float').columns): 
            plt.figure(count)
            plt.hist(col, data = place, range= tuple(place[col].quantile([.01,.99]).values))
            if ind == 0: 
                city = 'LA'
            else:
                city = 'SF'
            title = city+': "fixed" '+col
            plt.title(title)
            plt.savefig("files/plots/"+title+".png")
            count+=1
    #setting columns the same order and outputting data
    sf = sf[sorted(sf.columns)]
    la = la[sorted(la.columns)]
    sf.to_csv(output_sf)
    la.to_csv(output_la)

if __name__ == "__main__":
    os.chdir('/Users/pkim/Dropbox/Projects/SpringAccel')
    sns.set_style('whitegrid')
    sns.set_palette("husl")
    # Read in both LA and SF datasets
    file_path = 'files/july19/LA_2.csv'
    file_path_sf = 'files/july19/SF_2.csv'
    output_la = 'files/july19/LA_2b.csv'
    output_sf = 'files/july19/SF_2b.csv'
    data_fix(file_path,file_path_sf,output_la,output_sf)
