#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

pd.options.display.max_columns = None

try:
	os.chdir(os.path.join(os.getcwd(), '../2019 Spring/SpringAccel'))
	print(os.getcwd())
except:
	pass


# In[2]:


#Read in dataset 
#df = pd.read_csv('Airbnb_Listing_with_242Plus.csv',index_col = 0)
df = pd.read_csv('SanFranListing_With_Plus.csv',index_col = 1)



# In[3]:

df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()


# In[4]:
df.rename({'isPlus_y': 'isPlus'}, axis=1,inplace=True)

df.columns


# In[5]:


# creates new popluation where it's either Plus listings or Superhost listings
population = df[(df['isPlus'] == 1) | (df['host_is_superhost'] == 't')]


# In[6]:


population  = population.reset_index()#.drop(columns='index')


# In[7]:


# type corrections 
population['host_total_listings_count'] = population['host_total_listings_count'].astype(float)
population['host_listings_count']=population['host_listings_count'].astype(float)
population['accommodates']=population['accommodates'].astype(float)

rest = ['guests_included','minimum_nights','maximum_nights']
for i in rest: 
    population[i] = population[i].astype(float)
    
money = ['price','weekly_price','monthly_price','security_deposit','cleaning_fee','extra_people']
population['price']=population['price'].apply(lambda x: float("".join(list(x)[1:]).replace(',',"")))
population['weekly_price'] = population['weekly_price'].apply(lambda x: 0 if str(x)[:1] != "$" else x)
population['weekly_price']=population['weekly_price'].apply(lambda x: float(str(x)[1:].replace(',',"")) if len(str(x))>1 else float(0) )
money = ['monthly_price','security_deposit','cleaning_fee','extra_people']
for i in money:    
    population[i]=population[i].apply(lambda x: 0 if str(x)[:1] != "$" else x)
for i in money: 
    population[i]=population[i].apply(lambda x: float(str(x)[1:].replace(',',"")) if len(str(x))>1 else float(0) )


# In[19]:

# Making response rate into a float
population['host_response_rate']=population['host_response_rate'].apply(lambda x: float(str(x)[0:-1])/100 if str(x)[0:-1] != 'na' else np.NAN)


# In[20]:


population.info()

population['isPlus'] = population['isPlus'].apply(lambda x: 1 if x==1 else 0)
# In[21]:


# Looking at categorical columns 
cat = population.select_dtypes(include = 'object').copy()


# In[22]:
cat.head()

# In[23]:


for i in cat.columns:
    print(i,len(cat[i].unique()))


# In[24]:


for i in cat.columns:
    print(i,sum(cat[i].isnull()))


# In[25]:


cat.columns


# In[26]:
population['zipcode'] = population['zipcode'].astype('object')
# Columns that I'm keeping
cols = ['host_id', 'host_name', 'host_since',
        'host_response_time',
         'host_is_superhost',
       'host_neighbourhood','zipcode',
       'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
        'neighbourhood_cleansed', 'city', 'state',
        'market', 'smart_location', 
       'is_location_exact', 'property_type', 'room_type', 'bed_type',
       'amenities', 
        'instant_bookable',
       'cancellation_policy',
       'require_guest_profile_picture', 'require_guest_phone_verification']


# In[27]:

cat[cols].head()


# In[28]:

# Looking at # of unique values for each categorial variables
for i in cat[cols].columns:
    print(i,len(cat[i].unique()))

# In[29]

population['id']=population['id'].astype('float')
population['availability_30']=population['availability_30'].astype('int')
# Looking at Numerical data
num = population.select_dtypes(include = ['float','int']).copy()


# In[37]:

# Columns I'm keeping 
cols2 = ['id',
       'host_listings_count',
       'latitude', 'longitude', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'price', 'weekly_price',
       'monthly_price', 'security_deposit', 'cleaning_fee', 'guests_included',
       'extra_people', 'minimum_nights', 'maximum_nights', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'calculated_host_listings_count',
       'reviews_per_month', 'isPlus']


# In[38]:


num[cols2].head()


# In[39]:

# Looking at # of uniques in each cols
for i in cols2:
    print(i,len(num[i].unique()))


# In[40]:

# Final columns from Categorical and Numerical 
final_columns = cols + cols2


# In[41]:

# Setting population_final_columns as table
population_final_columns = population[final_columns]


# In[42]:

# Looking at NAN values
plt.figure(figsize=(20,12))
sns.heatmap(population_final_columns.isna())



# In[29]:


plt.figure(figsize=(20,12))
sns.heatmap(population_final_columns[population_final_columns['isPlus']==0].isna())

# In[32]:


just_plus = population_final_columns[population_final_columns['isPlus'] == 1]


# In[33]:


sns.scatterplot(x='number_of_reviews',y='review_scores_rating',
                data=just_plus)


# In[34]:


no_plus = population_final_columns[population_final_columns['isPlus'] != 1]


# In[35]:


see = ['accommodates',
       'bathrooms', 'bedrooms', 'beds','number_of_reviews',
       'review_scores_accuracy','review_scores_cleanliness', 
       'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 
       'reviews_per_month','minimum_nights', 'maximum_nights', 
       'availability_30',
       'availability_60', 'availability_90', 'availability_365', 
       'price', 'weekly_price',
       'monthly_price', 'security_deposit', 'cleaning_fee', 
       'guests_included',
       'extra_people']
for i in see: 
    print(i)
    print("just_plus",sum(just_plus[i])/len(just_plus))
    print("no_plus",sum(no_plus[i].dropna())/len(no_plus))
    print("all",sum(population_final_columns[i].dropna())/len(population_final_columns),'\n')
    
    
    


# In[36]:


g = sns.FacetGrid(no_plus                     
                  , row = 'host_is_superhost', col = 'isPlus')
g.map(plt.hist,"price")


# In[37]:


g = plt.figure()
sns.pairplot(population_final_columns.fillna(0,axis=0),hue='isPlus',vars=see[5:11],)


# In[38]:


sns.pairplot(population_final_columns.fillna("-1",axis=0),hue='isPlus',vars=see[18:23],)


# In[39]:


sns.pairplot(population_final_columns.fillna("-1",axis=0),hue='isPlus',vars=see[:5],)


# In[40]:


for i in population_final_columns.columns:
    print(i, str(sum(population_final_columns[i].isna())/len(population_final_columns)*100)[:5],'%')


# In[41]:


population_final_columns[population_final_columns['isPlus']==1]['number_of_reviews'].describe()


# In[42]:


population_final_columns[population_final_columns['isPlus']==0]['number_of_reviews'].describe()


# In[43]:


plt.figure(figsize=(20,12))
sns.heatmap(population[cols2].corr()) 


# In[43]:


population_final_columns.head()


# In[44]:

# To CSV
population_final_columns.to_csv('SanFranCleaner.csv')


# In[ ]:




