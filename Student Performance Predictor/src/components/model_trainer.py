import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.ensemble import StackingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and Test Input Data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()                
            }

            params = {
                "Decision Tree": {
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_depth':[3,5,7,9,11,13,15,17,20,None],
                    'min_samples_split':[2,3,4,5,6,7,8,9,10],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                   # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                   # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                   # 'learning_rate':[.1,.01,.05,.001],
                   # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                   # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{
                    'fit_intercept':[True,False],
                    'copy_X':[True, False]
                },
                "K-Neighbors Regressor":{
                    'n_neighbors':[1,2,3,4,5,6,7,8,9]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[.1,.01,.05,.001],
                    'iterations':[30,50,100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }                
            }

             # To evaluate all the models and return the best model score

            model_report:dict = evaluate_model(x_train,y_train,x_test,y_test,models,params)

             # To get best model score
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            best_model.fit(x_train,y_train)
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset")
            print(f"Best model found: {best_model_name} with r2 score: {best_model_score}")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info("Trained model saved")
            print("Saving model to:", self.model_trainer_config.trained_model_file_path)
            predicted = best_model.predict(x_test)

            r_score = r2_score(y_test,predicted)
            return r_score
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    from src.components.data_ingestion import DataIngestion

    try:
        # Run data ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        # Run transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # Train model
        trainer = ModelTrainer()
        score = trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"Model Training completed successfully with R2 Score: {score}")

    except Exception as e:
        raise CustomException(e, sys)

    