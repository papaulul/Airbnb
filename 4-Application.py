#%%
import os 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
am_i_local = "no"
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
file_path = 'SanFran-2.csv'
df_new_city = pd.read_csv(file_path,index_col = 0)
missing_cols = ['Air purifier','Alfresco bathtub','Barn','Boat','Brick oven','Bus','Camper/RV','Campsite','Castle','Chalet','Dome house','Farm stay','Houseboat','Hut','In-law','Lighthouse','Misting system','Mobile hoist','Other_Amen.1','Outdoor kitchen','Pool cover','Pool toys','Pool with pool hoist','Private gym','Private pool','Sauna','Ski-in/Ski-out','Tent','Tipi','Train','Treehouse','Yurt']
df_new_city[missing_cols] = df_new_city.loc[:,missing_cols]
df_new_city.fillna(False,inplace=True)
df_new_city.drop('In-law',axis=1,inplace=True)
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
# All important features will be put here 

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
    df_model['isPlus'] = df_model['isPlus'].astype('float')

    #%%
    important_cols = feats[0]
    print("Here are the important columns: ", important_cols)
jaccard(df_new_city)
#%%
f_cols = ['host_is_superhost','host_has_profile_pic','host_identity_verified','is_location_exact','instant_bookable','require_guest_profile_picture','require_guest_phone_verification','host_listings_count','latitude','longitude','accommodates','bathrooms','bedrooms','beds','price','monthly_price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','reviews_per_month','isPlus','number_of_unique_amenities','number_of_host_verifications','listing_count','Bathroom essentials','Breakfast table','Ceiling fan','En suite bathroom','Espresso machine','Full kitchen','Gas oven','HBO GO','Hot water kettle','Memory foam mattress','Mini fridge','Outdoor seating','Pillow-top mattress','Rain shower','Smart TV','Sun loungers','Walk-in shower','Netflix','a few days or more','within a day','within a few hours','within an hour','Apartment','Barn','Bed and breakfast','Boat','Boutique hotel','Bungalow','Bus','Cabin','Camper/RV','Campsite','Castle','Chalet','Condominium','Cottage','Dome house','Farm stay','Guest suite','Guesthouse','Hostel','Hotel','House','Houseboat','Hut','Lighthouse','Loft','Other_Amen.1','Resort','Serviced apartment','Tent','Tiny house','Tipi','Townhouse','Train','Treehouse','Villa','Yurt','Private room','Shared room','Couch','Futon','Pull-out Sofa','Real Bed','moderate','strict','strict_14_with_grace_period','super_strict_30','super_strict_60']

#%%
files = os.listdir('models')
for filename in files: 
    
    model =   pickle.load(open("models/"+filename, 'rb'))
    if 'f' in filename: 
        X = df_new_city[f_cols].drop('isPlus', axis=1 )
    else:
        X = df_new_city.drop('isPlus', axis=1)
    y = df_new_city['isPlus']
    predictions = model.predict(X)

    _,FP,FN,TP = confusion_matrix(y, predictions).ravel()
    print(filename, (float(TP))/(float(TP) + float(FP)+ float(FN)),"\nTP:", TP, "\nFN:",FN, "\nFP:",FP)
