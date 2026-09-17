import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_pickle
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            X_train,y_train,X_test,y_test=(
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
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # added all params for all models for hyperparameter tuning
            params = {
                "Decision Tree": {
                    "max_depth": [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                    "max_features": ["sqrt", None],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                },
                "CatBoosting Regressor": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.1],
                    "iterations": [100, 200],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.01, 0.1, 1.0],
                    "n_estimators": [50, 100, 200],
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                }
            }

            logging.info('Starting Hyperparameter tuning')

            # we are getting model report after hyperparameter tuning
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,params)

            # now selecting best model for this data
            best_model_name=max(model_report, key=lambda x: model_report[x]["score"])
            best_model=model_report[best_model_name]['best_model']
            best_model_score=model_report[best_model_name]['score']

            # we dont want the model if their score is less than .6 as they are just then
            if best_model_score<0.6:
                raise CustomException('No good model found',sys)

            logging.info('Best model found for both training and testing dataset')

            save_pickle(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
                
        except Exception as e:
            raise CustomException(e,sys)  
