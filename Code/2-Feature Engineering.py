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
input_path= 'files/SF-1b.csv'
output_path = 'files/SF-2.csv'
df = pd.read_csv(input_path, index_col='Unnamed: 0')
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

df = df_with_amenities.copy()

# In[3]:
# Finding Variables that have high correlation 
high_corr = pd.DataFrame(df.corr().abs().unstack()[df.corr().abs().unstack().sort_values(kind="quicksort")>.9]).reset_index()
print(high_corr[high_corr['level_0']!=high_corr['level_1']])

#%% 
# Removed variables with high correlations 
d_list = ['calculated_host_listings_count','number_of_amenities','Bath towel','Bedroom comforts','Body soap','Dishes and silverware','Toilet paper','Cooking basics','Dryer','Wide clearance to shower','amenities','availability_30','availability_365','availability_60','availability_90']
df.drop(d_list,inplace=True, axis=1)
df.head()

# In[11]:

# Finds index of where host_resposne_time is na and then set NAN value as unavaliable
newvar = list(df[(df['host_response_time'].isna())].reset_index()['index'])
for i in newvar:
    df.set_value(i, 'host_response_time','Unavaliable')

try: 
    df.rename({"listing_count_LA": "listing_count"}, axis=1,inplace=True)
except: 
    print("Your file is fine :) ")
# In[12]:
## Drop Column with high sparsity or other issues 
df.drop(['host_id','host_name','host_since','host_verifications','neighbourhood_cleansed','city','state','zipcode','market','smart_location','id','translation missing: en.hosting_amenity_49','translation missing: en.hosting_amenity_50','host_neighbourhood','weekly_price'],axis=1, inplace=True)
# In[13]:
cols = list(df.columns)
cols = list(filter(lambda x: 'Unamed:' not in x,cols))
df = df[cols]
#Verify that none of the rows with Plus listings have null 
# Dropping all rows with NA  cause doesn't affect Plus listings
# Changes object types into dummy variables 
items=list(df.select_dtypes('object').columns)
for cols in items:
    add = pd.get_dummies(df[cols],drop_first=True).astype('bool')
    df=pd.concat([df,add], axis=1)

df.rename({'Other': 'Other_Amen'}, axis=1,inplace=True)
# Changes an integer types into boolean 
# drop original categorical variables
df.drop(items,axis=1, inplace= True)

# In[19]:
# confirms that isPlus is boolean
df['isPlus'] = df['isPlus'].apply(lambda x: 1 if x == 1 else 0).astype('bool')
df.dropna(axis=0,inplace=True)
df.drop('',axis=1,inplace=True)
#%%
df.to_csv(output_path)



#%%
