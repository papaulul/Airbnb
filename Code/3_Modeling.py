# Imports
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time 
# Import modeling tools 
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, recall_score, accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from xgboost import XGBClassifier
start_time = time.time()

def scaled(X):
    # Iniate Standard Scaler
    ss = StandardScaler()
    # Making sure scaling floats
    to_scale = X.select_dtypes("float")
    # Getting transformed values
    scaled = ss.fit_transform(to_scale)
    # Saving fit for SF dataset 
    pickle.dump(ss, open("models/scaler.pkl","wb"))
    # Setting values to table
    X[to_scale.columns] = pd.DataFrame(scaled, columns = to_scale.columns)
    return X

def met(TP,FP,FN): 
    """
    Inputs: 
        TP: True Positives
        FP: False Postitives
        FN: False Negatives 
    Objective: 
        Give an 'error' to the model based on FN 
        formula: 1 - (TP+FP)/(TP+FP +FN) = FN/(TP+FP+FN)
    """
    return (float(FN))/(float(TP) + float(FP)+ float(FN))
def model_train_predict_matrix(model_type,X_train, X_test,y_train, y_test): 
    """
    Input: 
        model_type: Looks for string of different model short name I've given
    Output: 
        Fits model on X_train and y_train set
        makes prediction on X_test
        Will use that for the confusion matrix and will grab the True Positive and False Positive 
    """
    if model_type == 'bnb':
        model = bnb
    elif model_type == 'log':
        model = log
    elif model_type == 'rf':
        model = rf
    elif model_type == 'xgb':
        model = xgb
    else: 
        return [0,0]
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    _,FP,FN,TP = confusion_matrix(y_test, predictions).ravel()
    return [TP,FP,FN]
def rand_grid(params, model, X_train,y_train):
    """
    Input: 
    Parameters to test for models
    model variable 
    ---
    Output:
    model
    """
    grid = RandomizedSearchCV(model,params,n_iter=20)
    grid.fit(X_train,y_train)
    # Setting the model with best parameters 
    new_model = model.set_params(**grid.best_params_)
    print("Optimal Model Found")
    # Saving the model to save time for different runs
    output_file = "models/" + str(new_model).split("(")[0]+".sav"
    pickle.dump(new_model,open(output_file, "wb"))
    return new_model

def param_search(X,y):
    """
    Input:
    Dataframe with out targe
    Dataframe with target
    ---
    Output: 
    4 models with optimized parameters 
    """
    start_time = time.time()
    # All parameters to test for Random Grid Search 
    log_params = {
                'penalty':['l1','l2']
                }
    xgb_params = {
            "max_depth":[depth for depth in range(1,15,2)],
            "learning_rate": [lr/100 for lr in range(1,10,1)],
            "n_estimators": [est for est in range(1,2001,500)]
            }
    rf_params = {
            "max_depth":[depth for depth in range(1,15,2)],
            "max_features":[features for features in range(5,55,5)],
            "n_estimators": [est for est in range(1,2001,500)]
            }
    bnb_params = {
                'alpha': [alpha/10 for alpha in range(0,10,1)]
                }

    # Providing the datasets for the rand_grid function
    X_train, _, y_train, _ = train_test_split(X,
        y,train_size = 0.75,stratify=y, random_state=999)

    # Change this if you don't want to run the grid search
    first_run = True 
    if first_run: 
        log = rand_grid(log_params,LogisticRegression(solver="liblinear"),X_train,y_train)
        bnb = rand_grid(bnb_params,BernoulliNB(),X_train,y_train)
        rf = rand_grid(rf_params, RandomForestClassifier(),X_train,y_train)
        xgb = rand_grid(xgb_params,XGBClassifier(),X_train,y_train)
    else:
        log = pickle.load(open("models/LogisticRegression.sav","rb"))
        bnb = pickle.load(open("models/BernoulliNB.sav","rb"))
        rf = pickle.load(open("models/RandomForestClassifier.sav","rb"))
        xgb = pickle.load(open("models/XGBClassifier.sav","rb"))

    print("Parameter Search time in Seconds: ",time.time()-start_time)
    return log, bnb, rf, xgb 

def kfold(X,y):
    """
    Input: 
    Dataframe with out targe
    Dataframe with target
    """
    start_time = time.time()    
    # Picking Best Model Based on KFold
    skf = StratifiedKFold(5, shuffle= True,random_state = 999)
    # Dataframe to keep the score 
    scores = pd.DataFrame(columns=['bnb_met','log_met','rf_met','xgb_met'])
    # Will iterate through each fold and for each model 
    Iter = 1
    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X.loc[train_index],X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        for sets in [X_train,X_test,y_train,y_test]:
            # Removing any NA that might've snuck through
            sets.dropna(axis=0,inplace=True)
        for m_name in ['bnb','log','rf','xgb']: 
            # Getting TP,FP,FN for the metric
            TP,FP,FN = model_train_predict_matrix(m_name,X_train, X_test,y_train, y_test)
            # Calculating metric
            metric = met(TP,FP,FN)
            col = m_name +"_met"
            scores.loc[int(Iter),col] = metric
            print(col, str(Iter), metric)
        Iter += 1
    # convert everything to a float (sometimes it gets put in as integers)
    for i in list(scores.columns): 
        scores[i]=scores[i].astype('float')
    print(scores.describe())
    print("K-fold time in Seconds: ",time.time()-start_time)

def under_sampling(df_model):
    """
    Input: 
    Full dataset
    ---
    Output: 
    Dataset without target that is undersampled
    Target variables that are undersampled
    """
    # Looking into undersampling 
    np.random.seed(1)
    # Get number of Plus listings
    number_of_plus = sum(df_model['isPlus'])
    # the Index of non Plus Listings
    non_plus_index = df_model[df_model['isPlus'] == False].index
    # from Index of non Plus, randomly select wihtout replacement the number of Plus listings 
    random_index = np.random.choice(non_plus_index,number_of_plus,replace=False)
    # grab index of Plus listings
    plus_index = df_model[df_model['isPlus']==True].index
    # combine random and plus index and then sort
    under_sampled_indexes = np.sort(np.concatenate([plus_index,random_index]))
    # locate the rows from the indexes 
    under_sample = df_model.loc[under_sampled_indexes]
    # Set X and y and reset the index so that it works eaiser with the kfold function
    X_under = under_sample.drop('isPlus',1).reset_index().drop("index",1)
    y_under = under_sample['isPlus'].reset_index().drop("index",1)

    ###
    # Looking how undersampling performs against normal sampling
    ###
    X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(
                            X_under,y_under,test_size = 0.3, random_state = 0)

    X_train, X_test, y_train, y_test = train_test_split(X,
                            y,test_size = 0.3,stratify=y, random_state=999)
    
    # fitting log because it's fast to the under sampled set and evaluating
    log.fit(X_under_train,y_under_train)
    y_under_pred = log.predict(X_under_test)
    print("Undersampling Results: Recall, Accuracy, and Confusion Matrix")
    print(recall_score(y_under_test,y_under_pred))
    print(accuracy_score(y_under_test,y_under_pred))
    print(confusion_matrix(y_under_test,y_under_pred))
    # predicting on full set
    pred_all = log.predict(X_test)
    print("Undersampling on Normal Data Results: Recall, Accuracy, and Confusion Matrix")
    print(recall_score(y_test,pred_all))
    print(accuracy_score(y_test,pred_all))
    print(confusion_matrix(y_test,pred_all))
    # fitting log on full set as baseline
    log.fit(X_train,y_train)
    pred = log.predict(X_test)
    print("Normal Results: Recall, Accuracy, and Confusion Matrix")
    print(recall_score(y_test,pred))
    print(accuracy_score(y_test,pred))
    print(confusion_matrix(y_test,pred))
    return X_under,y_under

if __name__ == "__main__":
    os.chdir('/Users/pkim/Dropbox/Projects/SpringAccel')
    # Import from 2
    file_path = 'files/july19/LA_2b.csv'
    df = pd.read_csv(file_path,index_col = 0)
    # Removing superflous columns
    df_model = df.drop(['id','amenities','host_id'],1).copy()
    # Setting up the dataframes
    X = df_model.drop('isPlus',axis= 1)
    y = df_model['isPlus']
    
    # Scaling model
    X = scaled(X)
    # Setting models as more of a global variable
    log, bnb, rf, xgb = param_search(X,y)
    kfold(X,y)
    X_under, y_under = under_sampling(df_model)
    kfold(X_under, y_under)
    print("--- %s seconds ---" % (time.time() - start_time))
