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
        os.chdir(os.path.join(os.getcwd(), '../2019 Spring/SpringAccel'))
        print(os.getcwd())
    except:
        pass
from IPython.display import display

pd.options.display.max_columns = None

json_input = 'Airbnb/sf_img.json' 
csv_input = 'files/SF-1a.csv'

scraped = pd.read_json(json_input)
scraped = scraped['image_urls']
scraped = scraped.apply(lambda x: str(x).replace("['","").replace("']","")).values
current = pd.read_csv(csv_input)
current = current['picture_url']
#%%
to_save  = list(set(list(current)) ^ set(list(scraped)))

#%%
len(set(list(current)) ^ set(list(scraped))
)

#%%
pd.DataFrame(to_save, columns = ['picture_url']).to_csv('Airbnb/img.csv')

#%%
