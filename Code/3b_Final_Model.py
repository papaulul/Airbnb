import os
import pickle
import pandas as pd
import numpy as np
import time 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, recall_score
start_time = time.time()

def scaled(X):
    # Iniate Standard Scaler
    ss = pickle.load(open("models/scaler.pkl","rb"))
    # Making sure scaling floats
    to_scale = X.select_dtypes("float")
    # Getting transformed values
    scaled = ss.fit_transform(to_scale)
    # Setting values to table
    X[to_scale.columns] = pd.DataFrame(scaled, columns = to_scale.columns)
    return X

if __name__ == "__main__":
    os.chdir('/Users/pkim/Dropbox/Projects/SpringAccel')
    # Path to file
    file_path = 'files/july19/LA_2b.csv'
    df = pd.read_csv(file_path,index_col = 0)
    df_model = df.drop(['id','amenities','host_id'],1).copy()
    # Undersampling, keeping it the same as the previous file
    df_model = scaled(df_model)
    
    np.random.seed(1)
    number_of_plus = sum(df_model['isPlus'])
    non_plus_index = df_model[df_model['isPlus'] == False].index
    random_index = np.random.choice(non_plus_index,number_of_plus,replace=False)
    plus_index = df_model[df_model['isPlus']==True].index
    under_sampled_indexes = np.concatenate([plus_index,random_index])
    under_sample = df_model.loc[under_sampled_indexes]
    X_under = under_sample.drop('isPlus',1)
    y_under = under_sample['isPlus']
    # Keeping this the same as well
    X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(
        X_under,y_under,test_size = 0.3, random_state = 0)

    # XGB and Log performed the best with quickest times. Will look into them
    # Loading and fitting models
    xgb = pickle.load(open("models/XGBClassifier.sav","rb"))
    xgb.fit(X_under_train,y_under_train)
    log = pickle.load(open("models/LogisticRegression.sav","rb"))
    log.fit(X_under_train,y_under_train)
    # Making predictions on test sets and then full set
    predictions_xgb = xgb.predict(X_under_test)
    predictions_xgb_full = xgb.predict(df_model.drop("isPlus",1))
    predictions_log = log.predict(X_under_test)
    predictions_log_full = log.predict(df_model.drop("isPlus",1))
    # Performances 
    print("XGB Undersample Performance")
    print(classification_report(y_under_test,predictions_xgb))
    print(accuracy_score(y_under_test,predictions_xgb))
    print(recall_score(y_under_test,predictions_xgb))
    print(confusion_matrix(y_under_test,predictions_xgb))
    print("Logistic Regression Undersample Performance")
    print(classification_report(y_under_test,predictions_log))
    print(accuracy_score(y_under_test,predictions_log))
    print(recall_score(y_under_test,predictions_log))
    print(confusion_matrix(y_under_test,predictions_log))
    print("XGB Full Performance")
    print(classification_report(df_model['isPlus'],predictions_xgb_full))
    print(confusion_matrix(df_model['isPlus'],predictions_xgb_full))
    print("Logistic Regression Performance")
    print(classification_report(df_model['isPlus'],predictions_log_full))
    print(confusion_matrix(df_model['isPlus'],predictions_log_full))
    # XGB performed the best, will use for next file
    pickle.dump(xgb, open("models/XGB_Final_Model.sav","wb"))
    print("File runtime in Seconds: ",time.time()-start_time)
    