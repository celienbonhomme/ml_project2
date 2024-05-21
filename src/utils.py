"""Module for utils functions"""
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from time import time


def model_evaluation_lr(y_test, y_pred):
    mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
    r2 = round(r2_score(y_test, y_pred), 3)
    return {'mape': mape, 'rmse': rmse, 'r2': r2}


def VIF_calculation(X):
    """Calculate the Variance Inflation Factor (VIF) for each variable in the dataset"""
    VIF = pd.DataFrame()
    VIF["variable"] = X.columns
    VIF["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    VIF = VIF.sort_values('VIF', ascending=False).reset_index(drop = True)
    return(VIF)


def delete_multicollinearity(df, target_name, VIF_threshold):
    """Delete multicollinearity from the dataset"""
    X = df.drop(target_name, axis=1)
    start = time()
    VIF_mat = VIF_calculation(X)
    end = time()
    n_VIF = VIF_mat["VIF"][0]
    if (n_VIF <= VIF_threshold):
        print("There is no multicollinearity!")
    else:
        while (n_VIF > VIF_threshold):
            print(f'Dropped column {VIF_mat["variable"][0]} with VIF: {round(VIF_mat["VIF"][0], 1)} ({int(end - start)}s)')
            X = X.drop(VIF_mat["variable"][0], axis=1)
            start = time()
            VIF_mat = VIF_calculation(X)
            end = time()
            n_VIF = VIF_mat["VIF"][0]
    return X

def print_percentage_missing_values(df):
    nb_missing_values = df.isnull().sum().sum()
    percentage_missing_values = round(100*nb_missing_values / (df.shape[0] * df.shape[1]), 1)
    print(f'Missing values: {nb_missing_values} ({percentage_missing_values}%)')

def percentage_missing_values(df):
    nb_missing_values = df.isnull().sum().sum()
    percentage_missing_values = round(100*nb_missing_values / (df.shape[0] * df.shape[1]), 1)
    return percentage_missing_values