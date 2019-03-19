#%%
# Imports
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
## Import csv from 1b 
df = pd.read_csv('SanFranCleaner.csv', index_col='Unnamed: 0')
df.head()


#%%
# Calculations number of Amenities avaliable 
df['number_of_amenities'] = df['amenities'].apply(lambda x: len(x.split(','))) 
#%%
# Calculates unique number of Amenities avaliable 
df['number_of_unique_amenities'] = df['amenities'].apply(lambda x: len(set(x.split(',')))) 


#%%
# Calculates number of Verifications host provides 
df['number_of_host_verifications'] = df['host_verifications'].apply(lambda x: len(set(x.split(','))))

#%%
# Converting True/False columns to 1/0 
cols_need_convert = [
    'host_is_superhost', 'host_has_profile_pic','host_identity_verified','is_location_exact','instant_bookable','require_guest_profile_picture','require_guest_phone_verification'
]
for i in cols_need_convert:
    df[i] = df[i].apply(lambda x: 1 if x == 't' else 0)
#%%
# Dropping row with NAN value for State 
# only 1 row is missing 
df = df[df['state'].apply(lambda x: str(x).upper()) == 'CA']

#%%
# Converting host_since variable to datetime 
df['host_since'] = pd.to_datetime(df['host_since'])

#%%
# Finding # of listings each host have in the area
X = pd.DataFrame(df.groupby('host_id').size().reset_index()).rename(columns = {0: "listing_count"})
df = df.merge(X , on = 'host_id',how = 'left' )

#%%
# Expanding amenities to individual columns 
list_of_all_amenities = []
for i in df['amenities'].apply(lambda x: x[1:-1].split(",")):
    for j in i: 
        list_of_all_amenities.append(j)
list_of_all_amenities = list(set(list_of_all_amenities))
df['amenities'] = df['amenities'].apply(lambda x: x[1:-1].split(","))

#%%
list_of_all_amenities = sorted(list_of_all_amenities)
amend = pd.DataFrame(columns=list_of_all_amenities)
versus = df[['id','amenities']]
versus['amenities'] = versus['amenities'].apply(lambda x: sorted(x))

#%%
#%%
for i in list_of_all_amenities:
    if '"' in i:
        i = i.replace('"',"")
    versus[i] = False
    
#%%
for i,j in enumerate(versus['amenities']):
    for k in j:
        if '"' in k:
            k = k.replace('"',"")
        versus.set_value(i,k, True) 
#%%
versus.drop('amenities', axis=1, inplace = True)
#%%
versus.dropna(axis=0,inplace=True)
for i in versus.columns: 
    if sum(versus[i]) < 100:
        print(i,sum(versus[i]))

#%%
df_with_amenities = df.merge(versus, on = 'id', how = 'left')

#%%
df_with_amenities.to_csv('SanFranAmenitiespt2.csv')

#%%
