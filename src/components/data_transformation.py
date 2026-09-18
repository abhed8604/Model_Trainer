import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_preprocessor_obj(self,selected_feature):
        try:

            df=pd.read_csv('artifacts/train.csv')
            numerical_features = df.select_dtypes(include=['number']).columns.tolist()
            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # removing dependent feature from num feature
            if selected_feature in numerical_features:
                numerical_features.remove(selected_feature) 
            
            # to extract all uinque values from the categorical_features
            cat_feature_info={}
            for cat_feature in categorical_features:
                cat_feature_info[cat_feature]={
                    'name': cat_feature,
                    'values': df[cat_feature].dropna().unique().tolist()
                }
    
            # pipeline chains multiple steps of data transformation as one object
            num_pipeline=Pipeline(
                steps=[
                    # we use imputer to replace missing value with NaN or NULL
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
    
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(sparse_output=False, handle_unknown='ignore')), # reducing sparse output from OHE
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info('Created preprocessor object')
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                ]
            )

            return preprocessor,cat_feature_info

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_preprocessing(self,selected_feature):
        try:
        
            logging.info('Reading train and test data')
            
            train_df=pd.read_csv('artifacts/train.csv')
            test_df=pd.read_csv('artifacts/test.csv')

            # removing the rows with missing target columns as model cant be trained if target is NaN
            train_df = train_df.dropna(subset=[selected_feature])
            test_df = test_df.dropna(subset=[selected_feature])
            
            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj,cat_feature_info=self.get_preprocessor_obj(selected_feature)
    
            input_feature_train_df=train_df.drop(columns=[selected_feature])
            target_feature_train_df=train_df[selected_feature]
            
            input_feature_test_df=test_df.drop(columns=[selected_feature])
            target_feature_test_df=test_df[selected_feature]
    
            logging.info('Applying preprocessing objects on training and testing df')
            
            # now we are sending the train and test data to get preprocessed
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # we use c_ to add two arrays column to column
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info('Saving preprocessing object')

            # Save preprocessor 
            save_pickle(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Save categorical features info
            save_pickle(
                file_path=os.path.join('artifacts', 'cat_feature_info.pkl'),
                obj=cat_feature_info
            )
    
            return(
                train_arr,
                test_arr,
            )

        except Exception as e:
            raise CustomException(e,sys)
        