# Imports
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import display
from sklearn.linear_model import LinearRegression
import pickle
import time
import os
import warnings 
start_time = time.time()
warnings.filterwarnings('ignore')


def feature_engineering(city,list_of_amenities):
    ## Import csv from EDA 
    input_path= 'files/july19/'+city+'_1b.csv'
    df = pd.read_csv(input_path, index_col='Unnamed: 0')
    # SF seems to have the column "license" that LA did not have
    if city == "SF":
        df.drop("license",1,inplace=True)
    # Calculations number of Amenities avaliable 
    df['number_of_amenities'] = df['amenities'].apply(lambda x: len(x.split(','))) 
    # Calculates unique number of Amenities avaliable 
    df['number_of_unique_amenities'] = df['amenities'].apply(lambda x: len(set(x.split(',')))) 
    # Calculates number of Verifications host provides 
    df['number_of_host_verifications'] = df['host_verifications'].apply(lambda x: len(set(x.split(','))))
    # Change calendar update from categorical to number of days approximately
    calendar_update = { '1 week ago': 7,
                        '10 months ago': 300,
                        '11 months ago': 330,
                        '12 months ago': 365,
                        '2 days ago': 2,
                        '2 months ago': 61,
                        '2 weeks ago': 14,
                        '3 days ago': 3,
                        '3 months ago': 91,
                        '3 weeks ago': 21,
                        '4 days ago': 4,
                        '4 months ago': 121,
                        '4 weeks ago': 28,
                        '5 days ago': 5,
                        '5 months ago': 151,
                        '5 weeks ago': 35,
                        '6 days ago': 6,
                        '6 months ago': 181,
                        '6 weeks ago':42,
                        '7 months ago':211,
                        '7 weeks ago':49,
                        '8 months ago':241,
                        '9 months ago': 271,
                        'a week ago':7,
                        'today':0,
                        'yesterday':1
    }
    # Applying above dictionary
    df['calendar_updated'] = df['calendar_updated'].map(calendar_update).apply(lambda x: x if type(x) == float else 365)
    # Sorting amenities dictionary to a list. Making sure empty strings do not make the list
    df['amenities'] = df['amenities'].apply(lambda x: sorted([col.replace('"','').strip() for col in x[1:-1].split(",") if col != ""])) 
    # Temp table for amenities
    versus = df[['id','amenities']]
    # List of amenities was obtained from function: amenities. Will iterate through each and create columns for each
    # amenity 
    for i in list_of_amenities:
        versus[i] = False
    # Looping through each row and then through each list of amenities for the row.
    # If we find the ameniity, we set the value to true at the correct column
    for i,j in enumerate(versus['amenities']):
        for k in j:
            if '"' in k:
                k = k.replace('"',"")
            versus.set_value(i,k, True) 
    # Dropping any amenities. Also this created rows with all NAN, which we will drop as well
    versus.drop('amenities', axis=1, inplace = True)
    versus.dropna(axis=0,inplace=True)
    # Merging the tables by id 
    df = df.merge(versus, on = 'id', how = 'left')
    # initiating the function remove_corr to get rid of variables of high correlations
    df = remove_corr(df,city)
    ## Drop Column with high sparsity or other issues 
    df.drop(['host_verifications','translation missing: en.hosting_amenity_49','translation missing: en.hosting_amenity_50'],axis=1, inplace=True)
    # Changes object types into dummy variables that are not amenities or id
    items=[categorical for categorical in list(df.select_dtypes('object').columns) if categorical not in list_of_amenities and categorical != "amenities" and categorical != "id"]
    for cols in items:
        add = pd.get_dummies(df[cols],drop_first=True,prefix=cols).astype('bool')
        df=pd.concat([df,add], axis=1)
    # drop original categorical variables
    df.drop(items,axis=1, inplace= True)
    df['isPlus'] = df['isPlus'].apply(lambda x: 1 if x == 1 else 0).astype('bool')   
    return df

def missing_inpute(df, city):
    # Outputs csv file
    output_path = 'files/july19/'+city+'_2.csv'
    # The following is for situations I've found with the LA dataset that also coincide with the SF dataset
    # Took the average number of bedrooms for places with one bathrooms and filled NAN values with it
    bedroom_1 = df[df['bathrooms']==1]['bedrooms'].dropna().mean()
    bedroom_1_ind = df[(df['bathrooms']==1) & (df['bedrooms'].isna())].index
    df.at[bedroom_1_ind,'bedrooms'] = bedroom_1 

    # Took the average number of bedrooms for places with two bathrooms and filled NAN values with it
    bedroom_2 = df[df['bathrooms']==2]['bedrooms'].dropna().mean()
    bedroom_2_ind = df[(df['bathrooms']==2) & (df['bedrooms'].isna())].index
    df.at[bedroom_2_ind,'bedrooms'] = bedroom_2 

    # Took the average number of bedrooms for places with two bathrooms and filled NAN values with it
    bathroom_1 = df[df['bedrooms']==1]['bathrooms'].dropna().mean()
    bathroom_1_ind = df[(df['bedrooms']==1) & (df['bathrooms'].isna())].index
    df.at[bathroom_1_ind,'bathrooms'] = bathroom_1 
    
    # Filled any values not caught in the calendar update map with 365 which is the max I'm setting
    df.at[df[df['calendar_updated'].isna()].index,'calendar_updated'] = 365
    
    # Using linear regression to fill in the NA from the review columns
    df_no_na = df.copy().dropna(0)
    # Will use LA transformations as base then apply it SF rather than training it separately on SF
    if city =="LA":
        # Grab all review cols
        review_na_cols = [cols for cols in df_no_na.columns if "review_scores" in cols]
        # Getting rid of any features that are not numerical 
        object_cols = list(df_no_na.select_dtypes("object").columns)
        X = df_no_na.drop(review_na_cols+object_cols, 1).columns
        # Saving columns so it's easier for the SF dataset to get the same columns
        pickle.dump(X, open("files/transformations/cols_used.sav","wb"))
        # Setting up for the linear model
        X = df_no_na.drop(review_na_cols+object_cols, 1).as_matrix()
        # Iterate through each review column 
        for cols in review_na_cols:
            # Dependent variable
            y = df_no_na[cols].dropna().as_matrix()
            # all rows with na in the review column
            to_pred = df[df[cols].isna()].drop(review_na_cols+object_cols,1).as_matrix()
            lin = LinearRegression()
            lin.fit(X,y)
            # predicting rows with na in review
            pred = lin.predict(to_pred)
            # setting prediction to the rows
            df[cols].loc[df[cols].isna()] = pred
            # saving model for SF to use later 
            pickle.dump(lin, open("files/transformations/linear_missing_"+cols+".sav", 'wb'))
    else: 
        # SF follows about the same process but loading in the columns and models from LA
        review_na_cols = [cols for cols in df_no_na.columns if "review_scores" in cols]
        cols_used = pickle.load(open("files/transformations/cols_used.sav","rb"))
        for cols in review_na_cols:
            # find rows with na in review column and only using columns used in LA model
            to_pred = df[df[cols].isna()][cols_used].as_matrix()
            # loading LA review model
            lin = pickle.load(open("files/transformations/linear_missing_"+cols+".sav",'rb'))
            # Predicting and setting it to the na rows
            pred = lin.predict(to_pred)
            df[cols].loc[df[cols].isna()] = pred
    # making sure that all of new values fall within the right range
    for cols in review_na_cols:
        df[cols] = df[cols].apply(lambda x: 100 if x > 100 else int(x))
    df.to_csv(output_path)

def merge_columns(LA,SF):
    """
    LA is the dataframe for LA
    SF is the dataframe for SF 
    ----
    This is used to make sure that LA and SF end up with the same columns. Since there are some unique values in each
    city 
    """
    la = list(LA.columns)
    sf = list(SF.columns)
    la_col = [col for col in la if col not in sf]
    sf_col = [col for col in sf if col not in la]
    # Setting missing columns as False 
    for cols in la_col:
        SF[cols] = False 
    for cols in sf_col:
        LA[cols] = False 
    # Escape if columns are not the same
    if len(LA.columns) != len(SF.columns):
        print("ERROR")
        quit()
    return LA,SF

def amenities(LA,SF):
    """
    LA is the dataframe for LA
    SF is the dataframe for SF 
    ----
    Will get all unique amenities from the combination of both cities
    """
    list_of_all_amenities = []
    df_la = pd.read_csv(LA, index_col='Unnamed: 0')['amenities'].apply(lambda x: x[1:-1].replace('"',"").split(","))
    df_sf = pd.read_csv(SF, index_col='Unnamed: 0')['amenities'].apply(lambda x: x[1:-1].replace('"',"").split(","))
    df = pd.concat([df_la,df_sf])
    for i in df.values:
        for j in i: 
            j = j.strip()
            if j != "":
                list_of_all_amenities.append(j)
    list_of_all_amenities = sorted(list(set(list_of_all_amenities)))
    return list_of_all_amenities 


def remove_corr(df, city):
    # Finding Variables that have high correlation 
    # Just going to use LA as the baseline
    if city == "LA":
        # Finds variables that have correlation > .8 
        high_corr = pd.DataFrame(df.corr().abs().unstack()[df.corr().abs().unstack().sort_values(kind="quicksort")>.8]).reset_index()
        # Removing those where the variable is correlated with itself
        # Getting the Correlation Column
        d_list_cols = high_corr[high_corr['level_0']!=high_corr['level_1']].columns
        # Finding the index of the first variable with correlated values. To not remove both correlated variables
        final_index = high_corr[high_corr['level_0']!=high_corr['level_1']][d_list_cols[2]].drop_duplicates().index
        # list of all unique variables to remove that are too correlated
        d_list = list(high_corr[high_corr['level_0']!=high_corr['level_1']].loc[final_index]['level_1'].unique())
        # Dumping the correlated variables into a file for SF to use
        pickle.dump(d_list, open("files/july19/correlated_var.txt",'wb'))
    else: 
        # SF just read in the file from LA
        d_list = pickle.load(open("files/july19/correlated_var.txt","rb"))
    # Making sure that our target variable is not in the list to drop 
    if "isPlus" in d_list:
        d_list.remove("isPlus")
    # Dropping all correlated variables
    df = df.drop(d_list, axis=1)
    return df




if __name__ == "__main__":
    # Makes sure we're in the right directory
    os.chdir('/Users/pkim/Dropbox/Projects/SpringAccel')
    # Input path from EDA step
    input_path_LA= 'files/july19/LA_1b.csv'
    input_path_SF= 'files/july19/SF_1b.csv'
    # Get all unique amenities from both cities
    list_of_amenities = amenities(input_path_LA,input_path_SF)
    # various features are engineered like amenities as columns
    df_LA = feature_engineering('LA',list_of_amenities)
    df_SF = feature_engineering('SF',list_of_amenities)
    # merging the columns so that they are the same
    df_LA, df_SF = merge_columns(df_LA,df_SF)
    # Imputing all remaining NA values
    missing_inpute(df_LA,"LA")
    missing_inpute(df_SF,"SF")
    print("--- %s seconds ---" % (time.time() - start_time))
