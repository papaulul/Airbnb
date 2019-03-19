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
la1 = pd.read_json('Airbnb/latry1.json')
la2 = pd.read_json('Airbnb/latry2.json')
la3 = pd.read_json('Airbnb/latry3.json')
la4 = pd.read_json('Airbnb/latry4.json')
la5 = pd.read_json('Airbnb/latry5.json')
la6 = pd.read_json('Airbnb/latry6.json')
la7 = pd.read_json('Airbnb/latry7.json')
la8 = pd.read_json('Airbnb/latry8.json')
la9 = pd.read_json('Airbnb/latry9.json')
la10 = pd.read_json('Airbnb/latry10.json')
la11 = pd.read_json('Airbnb/latry11.json')

#%% 

#%%
# Create list of the name of all dataframes 
sanfrans = [la1,la2,la3,la4,la5,la6,la7,la8,la9,la10,la11]


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
final1 = la1.merge(la2, on = 'id', how = 'outer').merge(la3, on = 'id', how = 'outer').merge(la4,on='id',how='outer')
final2 = la5.merge(la6, on = 'id', how = 'outer').merge(la7, on = 'id', how = 'outer').merge(la8,on='id',how='outer')
final3 = la9.merge(la10, on = 'id', how = 'outer').merge(la11, on = 'id', how = 'outer')
#%%
# Repairing id to name and title 
final1['name']=final1['id'].apply(lambda x: x.split("|")[0])
final1['title']= final1['id'].apply(lambda x: x.split("|")[1])
final2['name']=final2['id'].apply(lambda x: x.split("|")[0])
final2['title']= final2['id'].apply(lambda x: x.split("|")[1])
final3['name']=final3['id'].apply(lambda x: x.split("|")[0])
final3['title']= final3['id'].apply(lambda x: x.split("|")[1])

#%%
# setting table as name and title only
final1 = final1[['name','title']]
final1=final1.rename({'name': 'host_name', 'title': 'name'},axis=1)
final2 = final2[['name','title']]
final2=final2.rename({'name': 'host_name', 'title': 'name'},axis=1)
final3 = final3[['name','title']]
final3=final3.rename({'name': 'host_name', 'title': 'name'},axis=1)

#%%
# Keep working here
final2 = final2.drop_duplicates()
final3 = final3.drop_duplicates()

#%%
final2.to_csv('pluslistings.csv')

#%%
# TO EXPORT 
san = pd.read_csv('Airbnb_Listing_with_242Plus.csv')

#%%
san['uid'] = san['host_name'] + "|" + san['name']
final3['uid'] = final3['host_name'] + "|" + final3['name']
#%%
final3['isPlus'] = 1
san = san.merge(final3[['uid','isPlus']], on = 'uid', how = 'left' )

#%%
sum(san['host_is_superhost'].apply(lambda x: 1 if x =='t' else 0 ))

#%%
san.to_csv('AirbnbWithMorePlus.csv')

#%%
