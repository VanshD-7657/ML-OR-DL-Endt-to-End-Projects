from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Configuration of the Data Ingestion Config

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import pymongo
import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split

# Reading from MongoDB Database
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def export_collection_as_dataframe(self):
        # Reading Data from MONGODB
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name 
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            # Converting in dataframe
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])
                
            # Replacing na with NULL values
            df.replace({"na":np.nan},inplace=True)

            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        feature_store_file_path = self.data_ingestion_config.feature_store_file_path
        # Creating Folders
        dir_path = os.path.dirname(feature_store_file_path)
        os.makedirs(dir_path,exist_ok=True)
        dataframe.to_csv(feature_store_file_path,index=False,header=True)
        return dataframe


    # Splitting data into Train_Test_Split
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            logging.info("Performed train test split on dataframe")
            train_set,test_set = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info("Exporting train and test file path")

            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)

            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info("Exported Training and Testing file path:")
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                 test_file_path=self.data_ingestion_config.testing_file_path)                          
            
            return dataingestionartifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)