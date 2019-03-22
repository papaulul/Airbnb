#%%
import os 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
am_i_local = "yes"
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
# Read in both LA and SF datasets
file_path = 'files/LA-2.csv'
file_path_sf = 'files/SF-2.csv'
la = pd.read_csv(file_path,index_col = 0)
sf = pd.read_csv(file_path_sf, index_col= 0)
output_la = 'files/LA-2b.csv'
output_sf = 'files/SF-2b.csv'

#%%
# Found missing columns from SF and LA but taking the difference of the set of columns
sf_missing_cols = list(set(la.columns) ^ set(sf.columns))

#['Alfresco bathtub','Apartment','Apple TV','Barn','Beach chairs','Boat','Breakfast bar','Brick oven','Bus','Camper/RV','Campsite','Castle','Ceiling fans','Chalet',"Chef's kitchen",'Dining area','Dome house','Earth house','Farm stay','Gas grill','Ice Machine','In-law','Infinity pool','Ironing Board','Misting system','Mobile hoist','Outdoor kitchen','Parking','Piano','Pond','Pool cover','Pool toys','Pool with pool hoist','Private gym','Private pool','Propane barbeque','Sauna','Security cameras','Tent','Timeshare','Tipi','Train','Treehouse','Wet bar','Wine storage','Yurt','luxury_moderate']

sf[sf_missing_cols] = sf.loc[:,sf_missing_cols]
sf.fillna(False,inplace=True)

la_missing_cols = list(set(la.columns) ^ set(sf.columns))
#['In-law','Timeshare'] 

la = pd.concat([la,pd.DataFrame(columns = la_missing_cols)])
la.fillna(False, inplace = True)
#%%
# Now that they have the same columns, going to run a feature importance on the all boolean columns
full = pd.concat([la,sf])
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances

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

    #%%
    important_cols = feats[0]
    print("Here are the important columns: ", important_cols)
    return important_cols
important_cols = jaccard(full)


#%%
# Going to drop any amenities that are not important 
places = [la,sf]
for place in places:
    for i in place.columns:
        try: 
            if place[i].dtype == 'bool' and i not in important_cols and i != 'isPlus':
                place.drop(i,axis=1,inplace = True)
        except:
                pass
    # Also dropping couple of others that were not important
    drop_list = ['host_has_profile_pic','host_identity_verified','host_is_superhost','instant_bookable','require_guest_phone_verification','require_guest_profile_picture','listing_count','is_location_exact']
    place.drop(drop_list, axis=1, inplace=True)

#%%
## Latitude/longitude idea: Instead of lat/long, make it to distance from average
for place in places: 
    avg_longitude = place['longitude'].mean()
    avg_latitude = place['latitude'].mean()
    place['d_long'] = place['longitude'].apply(lambda x: abs(x - avg_longitude))
    place['d_lat'] = place['latitude'].apply(lambda x: abs(x-avg_latitude))
    place.drop(['latitude','longitude'],axis=1,inplace=True)
    # Was lazy, so I decided here was the place to change interger columns to float
    place[place.select_dtypes('int').columns] = place.select_dtypes('int').astype('float')
#%%

#%%
# Look at all float columns
for ind,place in enumerate(places):
    for j,i in enumerate(sorted(place.select_dtypes('float').columns)): 
        plt.figure(j+10**ind)
        plt.hist(i, data = place)
        if ind == 0: 
            city = 'LA'
        else:
            city = 'SF'
        plt.title(city+": "+i)
        print('\n')
# Pulled any columns that were right skewed 
# Except those that involve price. I didn't want to touch those columns
right_skewed = ['reviews_per_month','number_of_reviews','guests_included','extra_people','d_long','d_lat','beds','bedrooms','bathrooms','accommodates']

import math
for place in places: 
    for skew in right_skewed:
        # trying to fix right_skewed with either sqrt or log
        #place[skew+"sqrt"] = place[skew].apply(lambda x: math.sqrt(abs(x)))
        place[skew+"log"] = place[skew].apply(lambda x: math.log(abs(x)) if abs(x) > 0 else 0)
    place.drop(right_skewed,axis=1,inplace=True)
##%% 
#%%
# checking to see if we fixed the skewed data
for ind,place in enumerate(places):
    for j,i in enumerate(sorted(place.select_dtypes('float').columns)): 
        plt.figure(j+10**ind)
        plt.hist(i, data = place)
        if ind == 0: 
            city = 'LA'
        else:
            city = 'SF'
        plt.title(city+": "+i)
        print('\n')
#%%
#setting columns the same order and outputting data
sf = sf[sorted(sf.columns)]
la = la[sorted(la.columns)]
sf.to_csv(output_sf)
la.to_csv(output_la)


#%%


#%%


#%%
