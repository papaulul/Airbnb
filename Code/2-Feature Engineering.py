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
input_path= 'files/LA-1b.csv'
output_path = 'files/LA-2.csv'
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
#Expanding amenities to individual columns 
list_of_all_amenities = []
for i in df['amenities'].apply(lambda x: x[1:-1].replace('"',"").split(",")):
    for j in i: 
        j = j.strip()
        if j != "":
            list_of_all_amenities.append(j)
list_of_all_amenities = sorted(list(set(list_of_all_amenities)))

#%%
df['amenities'] = df['amenities'].apply(lambda x: sorted([col.replace('"','').strip() for col in x[1:-1].split(",") if col != ""])) 
versus = df[['uid','amenities']]

#%%
for i in list_of_all_amenities:
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
    if i !='uid':
        if sum(versus[i]) < 100:
            print(i,sum(versus[i]))

#%%
df_with_amenities = df.merge(versus, on = 'uid', how = 'left')

#%%

df = df_with_amenities.copy()

#%%
# Finding Variables that have high correlation 
high_corr = pd.DataFrame(df.corr().abs().unstack()[df.corr().abs().unstack().sort_values(kind="quicksort")>.8]).reset_index()
#%% 
# Removed variables with high correlations 
d_list = list(high_corr[high_corr['level_0']!=high_corr['level_1']]['level_0'].unique())
df.drop(d_list,inplace=True, axis=1)
df.head()

#%%
## Drop Column with high sparsity or other issues 
df.drop(['host_verifications','translation missing: en.hosting_amenity_49','translation missing: en.hosting_amenity_50'],axis=1, inplace=True)

#%%
#Verify that none of the rows with Plus listings have null 
# Dropping all rows with NA  cause doesn't affect Plus listings
# Changes object types into dummy variables 
items=[categorical for categorical in list(df.select_dtypes('object').columns) if categorical not in list_of_all_amenities and categorical != "amenities" and categorical != "uid"]
for cols in items:
    add = pd.get_dummies(df[cols],drop_first=True,prefix=cols).astype('bool')
    df=pd.concat([df,add], axis=1)
#%%
# Changes an integer types into boolean 
# drop original categorical variables
df.drop(items,axis=1, inplace= True)

# In[19]:
# confirms that isPlus is boolean
df['isPlus'] = df['isPlus'].apply(lambda x: 1 if x == 1 else 0).astype('bool')

#%%
plt.scatter(df['bathrooms'],df['bedrooms'])

print(df[df['bedrooms'].isna()][['bedrooms','bathrooms']])

print(df[df['bathrooms'].isna()][['bedrooms','bathrooms']])

bedroom_1 = df[df['bathrooms']==1]['bedrooms'].dropna().mean()
bedroom_1_ind = df[(df['bathrooms']==1) & (df['bedrooms'].isna())].index
bedroom_2 = df[df['bathrooms']==2]['bedrooms'].dropna().mean()
bedroom_2_ind = df[(df['bathrooms']==2) & (df['bedrooms'].isna())].index
bathroom_1 = df[df['bedrooms']==1]['bathrooms'].dropna().mean()
bathroom_1_ind = df[(df['bedrooms']==1) & (df['bathrooms'].isna())].index

for ind in bedroom_1_ind:
    df.at[ind,'bedrooms'] = bedroom_1 

for ind in bedroom_2_ind:
    df.at[ind,'bedrooms'] = bedroom_2 
    
for ind in bathroom_1_ind: 
    df.at[ind,'bathrooms'] = bathroom_1 


#%%
from sklearn.linear_model import LinearRegression
#%%
df_no_na = df.copy().dropna(0)
review_na_cols = [cols for cols in df_no_na.columns if "review_scores" in cols]
object_cols = list(df_no_na.select_dtypes("object").columns)
X = df_no_na.drop(review_na_cols+object_cols, 1).as_matrix()
for cols in review_na_cols:
    y = df_no_na[cols].dropna().as_matrix()
    to_pred = df[df[cols].isna()].drop(review_na_cols+object_cols,1).as_matrix()
    lin = LinearRegression()
    lin.fit(X,y)
    pred = lin.predict(to_pred)
    df[cols].loc[df[cols].isna()] = pred

#%%
for cols in review_na_cols:
    df[cols] = df[cols].apply(lambda x: 100 if x > 100 else int(x))
#%%
df.to_csv(output_path)



#%%
