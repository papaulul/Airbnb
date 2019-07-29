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
        os.chdir(os.path.join(os.getcwd(), '../SpringAccel'))
        print(os.getcwd())
    except:
        pass
from IPython.display import display

pd.options.display.max_columns = None

csv_input = 'files/listings_LA_7_8.csv'
csv_output = 'files/july19/LA_1a.csv'
plus_input1 = 'Airbnb/airbnb_plus.csv'
plus_input2 = 'Airbnb/airbnb_plus2.csv'
#%% 
plus1 = pd.read_csv(plus_input1)
plus2 = pd.read_csv(plus_input2)

#%%
plus = plus1.merge(plus2, on = 'url',how = 'outer')

#%%
plus['url'] = plus['id'].apply(lambda x: "https://www.airbnb.com/rooms/"+x)

#%%
# Reading in file with all pulled data 
# Data is from InsideAirbnb
all_listings = pd.read_csv(csv_input)

#%%
# Setting the isPlus column with the merge
plus['isPlus'] = 1
all_listings = all_listings.merge(plus[['url','isPlus']], left_on = 'listing_url',right_on="url", how = 'left' ).drop("url",axis=1)
all_listings['isPlus'] = all_listings['isPlus'].apply(lambda x: 1 if x == 1 else 0)
#%%
# Seeing how many plus listings survived the merge
sum(all_listings['host_is_superhost'].apply(lambda x: 1 if x =='t' else 0 ))

population = all_listings[(all_listings['isPlus'] == 1) | (all_listings['host_is_superhost'] == 't')]

#%%
# creates file for the next state 
population.to_csv(csv_output)

