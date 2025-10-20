import pandas as pd
import sys 
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from data_ingestion import DataIngestion
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
import dill
from src.utils import save_object

# creating the object of DataIngestion
data_ingestion = DataIngestion()

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function use to transform the data
        '''
        try:
            numerical_faetures = ['reading score', 'writing score']
            categorical_features = ['gender',
            'race/ethnicity',
            'parental level of education',
            'lunch',
            'test preparation course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),#handle the missing value with median value
                    ('scalar',StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('OneHotEncode',OneHotEncoder()),
                    ('scalar',StandardScaler()),
                ]
            )
            logging.info("Numericals columns Standard scaling completed")
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_faetures),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            preprocessor = self.get_data_transformation_object()

            target_colum = 'math_score'
            numerical_faetures = ['reading score', 'writing score']
            input_features_train_df = train_df.drop(columns=[target_colum],axis=1)
            target_feature_train_df = train_df[target_colum]

            input_features_test_df = test_df.drop(columns=[target_colum],axis=1)
            target_feature_test_df = test_df[target_colum]
            logging.info("Appling the preprocessing in the both train and test dataset")

            input_features_train_arr = preprocessor.fit_transform(input_features_train_df) 
            target_feature_test_arr = preprocessor.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                target_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("Saving preprocessing object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor,
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e: 
            raise CustomException(e,sys)




