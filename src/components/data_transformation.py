import pandas as pd
import sys 
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    # file path where pkl file will save

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

# this finction return the object while helps to encode,hanldle missing value and scaling the data
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
                    ('scalar',StandardScaler()), #scaling the data
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")), # handling the missing value with mode
                    ('OneHotEncode',OneHotEncoder()),
                    ('scalar',StandardScaler(with_mean=False)),
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
            train_df = pd.read_csv(train_path) # reading train dataset
            test_df = pd.read_csv(test_path) # reading test dataset
            logging.info("Read train and test data")

            preprocessor = self.get_data_transformation_object()
            target_colum = 'math score'
            input_features_train_df = train_df.drop([target_colum],axis=1) # independent varaible for train
            target_feature_train_df = train_df[target_colum] # target varaible for train

            input_features_test_df = test_df.drop([target_colum],axis=1)# Independent varaible for test
            target_feature_test_df = test_df[target_colum] # target varaible for test
            logging.info("Appling the preprocessing in the both train and test dataset")

            # return array after transforming the dataset(train)
            input_features_train_arr = preprocessor.fit_transform(input_features_train_df)
            # return array after transforming the test dataset 
            target_feature_test_arr = preprocessor.transform(input_features_test_df)

            # this is the shortcut for concating the two array column wise
            train_arr = np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                target_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("Saving preprocessing object")
            
            # this method save the pkl file in os.path.join('artifacts','preprocessor.pkl')
            # --> utils file (root dir)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor,
            )
            # it return the train array and test array(for model traning) along with preprocessor file 
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e: 
            raise CustomException(e,sys)




