import os 
import sys 
import numpy as np
import pandas as pd 
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score


# for saving the pkl file 
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    


# for trainig and evaluating the models
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for m in list(models):
            model = models[m]
            model.fit(x_train,y_train)

            # y_train_pred = model.predict(x_test)
            y_test_pred = model.predict(x_test)

            # # for train data
            # score_train = r2_score(y_train,y_train_pred)
            # mae_train = mean_absolute_error(y_train,y_train_pred)
            # mse_train = mean_squared_error(y_train,y_train_pred)
            # rmse_train = root_mean_squared_error(y_train,y_train_pred)

            # for test data
            score_test = r2_score(y_test,y_test_pred)
            # mea_test = mean_absolute_error(y_test,y_test_pred)
            # mse_test = mean_squared_error(y_test,y_test_pred)
            # rmse_test = root_mean_squared_error(y_test,y_test_pred)
            report[m] = score_test

            return report
    except Exception as e:
        raise CustomException(e,sys)