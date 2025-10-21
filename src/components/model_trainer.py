import os
import sys 
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion


from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')
# saving the pkl file in artifacts

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Spliting trainig and test data")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],#selecting the all column except last
                train_arr[:,-1], #selecting last column
                test_arr[:,:-1], # for train data
                test_arr[:,-1] # for test data
            )
            models = {
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "SupportVectorRegressor":SVR(),
                "AdaboostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "XGBRFRegressor":XGBRFRegressor(),
                "CatBoostRegressor":CatBoostRegressor()
                }
            logging.info("Model trainig started")
            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_r2score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_r2score)]

            best_model = models[best_model_name]

            if best_r2score < 0.6: # this is the threshold value
                raise CustomException("No best model found")
            logging.info("Best model is founded")

            save_object(
                file_path = self.model_trainer_config.train_model_file_path,
                obj = best_model
            )
            '''
            Return the model name and r2score of best model
            '''
            predicated = best_model.predict(x_test)
            model_r2_score = r2_score(y_test,predicated)
            return(
                best_model,
                model_r2_score
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion()
    dt_taransformer = DataTransformation()
    train_arr,test_arr,preprocessor_path = dt_taransformer.initiate_data_transformation(train_path,test_path)
    model_trainer = ModelTrainer()
    score = model_trainer.initate_model_trainer(train_arr,test_arr,preprocessor_path)
    print(score)
            
