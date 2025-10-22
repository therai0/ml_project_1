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
            dill.dump(obj,file_obj) # saving file in bytes file 
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

            # for test data
            score_test = r2_score(y_test,y_test_pred)
            report[m] = score_test

            return report
    except Exception as e:
        raise CustomException(e,sys)
    

# load the pkl file and return the object 
def load_model_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

