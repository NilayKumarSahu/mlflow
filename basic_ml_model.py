import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import argparse


def get_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    #reading the data as df
    try:
        df = pd.read_csv(url,sep =";") # ; used as a separator
        return df
    except Exception as e:
        raise e

def evaluate(y_true, y_pred, pred_prob):
    try:
        '''mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.mean(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)'''
        accuracy = accuracy_score(y_true, y_pred)
        rc_auc_score = roc_auc_score(y_true, pred_prob, multi_class='ovr')
        return accuracy, rc_auc_score
        
    except Exception as e:
        raise e

def main(n_estimators, max_depth):
    df = get_data()
    try:
        train, test = train_test_split(df)
        x_train = train.drop(['quality'], axis = 1)
        x_test = test.drop(['quality'], axis = 1)

        y_train = train[['quality']]
        y_test = test[['quality']]
        #model training
        '''lr = ElasticNet()
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)'''
        with mlflow.start_run():
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            pred_prob = rf.predict_proba(x_test)
    

            #evaluate the model
            #mse, mae, rmse, r2 = evaluate(y_test, y_pred)
            accuracy, rc_auc_score = evaluate(y_test, y_pred, pred_prob)

            mlflow.log_param('n_estimators', n_estimators)
            mlflow.log_param('max_depth', max_depth)
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('roc_auc_score', rc_auc_score)

            #mlflow model logging
            mlflow.sklearn.log_model(rf, "random_forest_model")

            #print(f"Mean Squared Error{mse},Mean Absolute Error{mae},Root Mean squared Error{rmse},r2_score{r2}")
            print(f"Accuracy Score{accuracy}, ruc_auc_score{rc_auc_score}")

    except Exception as e:
        raise e
 



if __name__== '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--n_estimators','-n', default=50, help = str, type = int)
    args.add_argument('--max_depth','-m', default=5, help = str, type = int)
    parse_args = args.parse_args()
    main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)