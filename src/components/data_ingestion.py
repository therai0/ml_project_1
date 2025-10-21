import sys
import os 
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts',"train.csv")
    test_data_path = os.path.join('artifacts',"test.csv")
    raw_data_path = os.path.join('artifacts',"raw.csv")
    ''' 
    This are the path(on base dir) where this file are created and save under the folder name artifacts.
    '''

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        '''
        Creating the object 
        '''

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion conponent or method")
        try:
            # reading the dataset
            df= pd.read_csv(os.path.join('notebook/data/StudentsPerformance.csv'))
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            # index false -> don't write the index column
            # header true -> it write the column name or header row
            logging.info("Train test splited")

            # spliting dataset into train and test
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")
            # it return the path
            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path,
            )
#    When we crate the object and call this initiate_data_ingestoin method it will return the 
# two value that is train data path and test data path
        except Exception as e:
            raise CustomException(e,sys)

'''
Here in every line there is a loggin message to know about the where code is getting excuted
logging.info()

Importing the CustomException

'''

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

