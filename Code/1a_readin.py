#%%
""" 
Reads in the json files from my spider
"""
import json 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from IPython.display import display
import time
start_time = time.time()

pd.options.display.max_columns = None
def readin(file_name, city):
    # Reading in full file from Inside Airbnb
    csv_input = os.path.join(os.getcwd(),'files/' + file_name)
    # Output file name 
    csv_output = os.path.join(os.getcwd(),'files/july19/'+ city.upper() + '_1a.csv')
    # Input from Scrapes. Will contain Host name, Listing Title, and URL of plus listings
    # Multiple runs were needed because some of the plus listings were not captured in the 
    # first run
    plus_input1 = os.path.join(os.getcwd(),'Airbnb/airbnb_plus_'+city+'.csv')
    plus_input2 = os.path.join(os.getcwd(),'Airbnb/airbnb_plus_'+city+'2.csv')

    # Merging the two Scrape files
    plus1 = pd.read_csv(plus_input1)
    plus2 = pd.read_csv(plus_input2)    
    plus = plus1.merge(plus2, on = 'url',how = 'outer')

    # url is the same as the one in the main table, but it contains /plus. Will remove to match it as the key
    plus['url'] = plus['url'].apply(lambda x: x.replace("/plus",""))

    
    # Reading in file with all pulled data 
    # Data is from InsideAirbnb
    all_listings = pd.read_csv(csv_input)

    
    # Setting the isPlus column with the merge
    plus['isPlus'] = 1
    all_listings = all_listings.merge(plus[['url','isPlus']], left_on = 'listing_url',right_on="url", how = 'left' )
    all_listings['isPlus'] = all_listings['isPlus'].apply(lambda x: 1 if x == 1 else 0)
    
    # Seeing how many plus listings survived the merge
    print(sum(all_listings['host_is_superhost'].apply(lambda x: 1 if x =='t' else 0 )))

    # Population will consist of those with Plus listings or superhost. This is because superhosts are the only
    # ones that can have plus listings
    population = all_listings[(all_listings['isPlus'] == 1) | (all_listings['host_is_superhost'] == 't')]

    # creates file for the next state 
    population.to_csv(csv_output)

if __name__ == "__main__":
    # Makes sure we're the correct directory
    os.chdir('/Users/pkim/Dropbox/Projects/SpringAccel')
    print(os.getcwd())
    # Reading LA and SF
    readin('listings_LA_7_8.csv', 'la')
    readin('listings_SF_7_8.csv', 'sf')
    print("--- %s seconds ---" % (time.time() - start_time))
