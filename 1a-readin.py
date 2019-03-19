#%% 
""" 
Reads in the json files from my scrape
"""
import json 
import numpy as np 
import pandas as pd 
import os
try:
        # So that VS code works 
	os.chdir(os.path.join(os.getcwd(), '../2019 Spring/SpringAccel'))
	print(os.getcwd())
except:
	pass


#%% 
# Read in the json files as dataframes 
place = pd.read_json()

#%%
# Create list of the name of all dataframes 
all_places = [place]

#%%
# Remove json brackets and such for both columns 
for i in all_places: 
    i['title'] = i['title'].apply(lambda x: str(x)[2:-2])
    i['name'] = i['name'].apply(lambda x: str(x)[2:].split(",")[0])



#%%
# Fill in empty as NAN 
for i in all_places: 
    for j in ['name','title']:
        i[j] = i[j].apply(lambda x: np.NAN if len(x) == 0 else x)

#%%
# seeing how many NAN we are dealing with for each column 
for k,i in enumerate(all_places): 
    for j in ['name','title']:
        print(k,str(j),sum(i[j].isna()))
#%%
# Creating unique id
for i in all_places:
    i['id'] = i['name']+"|" + i['title']
#%%
# removing NAN before merge
for i in all_places:
    i.dropna(inplace= True)
#%% 
# Ideally, merge all until you get final amount you are looking for
final = sanfran1.merge(sanfran2, on = 'id', how = 'outer')
#%%
# Repairing id to name and title 
final['name']=final['id'].apply(lambda x: x.split("|")[0])
final['title']= final['id'].apply(lambda x: x.split("|")[1])

#%%
# setting table as name and title only
final = final[['name','title']]
final=final.rename({'name': 'host_name', 'title': 'name'},axis=1)
#%%
# Reading in file with all pulled data 
# Data is from InsideAirbnb
all_listings = pd.read_csv('')

#%%
# Creating a unique id so i can perform a left join 
all_listings['uid'] = all_listings['host_name'] + "|" + all_listings['name']
final['uid'] = final['host_name'] + "|" + final['name']
#%%
# Setting the isPlus column with the merge
final['isPlus'] = 1
all_listings = all_listings.merge(final[['uid','isPlus']], on = 'uid', how = 'left' )

#%%
# Seeing how many plus listings survived the merge
sum(all_listings['host_is_superhost'].apply(lambda x: 1 if x =='t' else 0 ))

#%%
# creates file for the next state 
all_listings.to_csv('')

#%%
