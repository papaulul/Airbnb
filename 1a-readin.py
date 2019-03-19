#%% 
import json 
import numpy as np 
import pandas as pd 
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../2019 Spring/SpringAccel'))
	print(os.getcwd())
except:
	pass


#%% 
# Read in the json files as dataframes 
sanfran1 = pd.read_json('Airbnb/sanfran.json')
sanfran2 = pd.read_json('Airbnb/sanfran2.json')
sanfran3 = pd.read_json('Airbnb/sanfran3.json')

#%%
# Create list of the name of all dataframes 
sanfrans = [sanfran1, sanfran2, sanfran3]

#%%
# Remove json brackets and such for both columns 
for i in sanfrans: 
    i['title'] = i['title'].apply(lambda x: str(x)[2:-2])
    i['name'] = i['name'].apply(lambda x: str(x)[2:].split(",")[0])



#%%
# Fill in empty as NAN 
for i in sanfrans: 
    for j in ['name','title']:
        i[j] = i[j].apply(lambda x: np.NAN if len(x) == 0 else x)

#%%
# seeing how many NAN we are dealing with for each column 
for k,i in enumerate(sanfrans): 
    for j in ['name','title']:
        print(k,str(j),sum(i[j].isna()))
#%%
# Creating unique id
for i in sanfrans:
    i['id'] = i['name']+"|" + i['title']
#%%
# removing NAN before merge
for i in sanfrans:
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
san = pd.read_csv('SanFranlistings.csv')

#%%
san['uid'] = san['host_name'] + "|" + san['name']
final['uid'] = final['host_name'] + "|" + final['name']
#%%
final['isPlus'] = 1
san = san.merge(final[['uid','isPlus']], on = 'uid', how = 'left' )

#%%
sum(san['host_is_superhost'].apply(lambda x: 1 if x =='t' else 0 ))

#%%
san.to_csv('SanFranListing_With_Plus.csv')

#%%
