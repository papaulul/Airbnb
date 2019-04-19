#%% 
""" 
Reads in the json files from my scrape
"""
import json 
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

json_input = 'Airbnb/sf.json' 
csv_input = 'files/SanFran_3_19.csv'
csv_output = 'files/SF-1a.csv'
#%% 
# Read in the json files as dataframes 
place = pd.read_json(json_input)

#%%
# Remove json brackets and such for both columns 
place['title'] = place['title'].apply(lambda x: str(x)[1:-1].replace("'","").replace('"',''))
place['name'] = place['name'].apply(lambda x: str(x).split(",")[1].replace("'","").replace("]","").strip() if len(x)> 0 else '' )



#%%
# Fill in empty as NAN 
for j in ['name','title']:
    place[j] = place[j].apply(lambda x: np.NAN if len(x) == 0 else x)

#%%
# seeing how many NAN we are dealing with for each column 
for j in ['name','title']:
    print(str(j),sum(place[j].isna()))
#%%
# Creating unique id
place['uid'] = place['name']+"|" + place['title']
#%%
# removing NAN before merge
place.dropna(axis= 0, inplace= True)
place.drop_duplicates(inplace=True)
#%% 
place=place.rename({'name': 'host_name', 'title': 'name'},axis=1)
#%%
# Reading in file with all pulled data 
# Data is from InsideAirbnb
all_listings = pd.read_csv(csv_input)

#%%
# Creating a unique id so i can perform a left join 
all_listings['uid'] = all_listings['host_name'] + "|" + all_listings['name']
place['uid'] = place['host_name'] + "|" + place['name']
#%%
# Setting the isPlus column with the merge
place['isPlus'] = 1
all_listings = all_listings.merge(place[['uid','isPlus']], on = 'uid', how = 'left' )

#%%
# Seeing how many plus listings survived the merge
sum(all_listings['host_is_superhost'].apply(lambda x: 1 if x =='t' else 0 ))

population = all_listings[(all_listings['isPlus'] == 1) | (all_listings['host_is_superhost'] == 't')]

#%%
# creates file for the next state 
population.to_csv(csv_output)

