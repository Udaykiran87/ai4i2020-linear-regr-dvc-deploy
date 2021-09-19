# load and train and test file
# train algo
# save the metrices, params

import os
import warnings
import sys
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pandas_profiling import ProfileReport
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed
import time
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV, ElasticNet , ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from get_data import read_params
import argparse
import joblib
import json

def eval_matrics(actual,pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = config["base"]["target_col"].replace(" ","_")

    train = pd.read_csv(train_data_path, sep = ",")
    test = pd.read_csv(test_data_path, sep = ",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis = 1)
    test_x = test.drop(target, axis = 1)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)

    lr.fit(train_x, train_y)

    predicted_air_temp = lr.predict(test_x)

    (rmse, mae, r2) = eval_matrics(test_y, predicted_air_temp)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

#####################################################

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent = 4)

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio
        }
        json.dump(params, f, indent=4)
#####################################################

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(lr, model_path)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)