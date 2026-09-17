import os
import sys
import pandas as pd
from src.logger import logging
from src.exceptions import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionPath:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_path=DataIngestionPath()

    def initiate_data_ingestion(self,file):
        logging.info('Entered the data ingestion')
        try:
            df=pd.read_csv(file)
            
            os.makedirs(os.path.dirname(self.ingestion_path.train_data_path),exist_ok=True)
    
            logging.info('Doing the train test split')
            train_set,test_set=train_test_split(df,test_size=0.25,random_state=7)
    
            train_set.to_csv(self.ingestion_path.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_path.test_data_path,index=False,header=True)

            logging.info('Train test split done')

            return df.columns.to_list()

        except Exception as e:
            logging.info('Error occured while doing the train test split')
            raise CustomException(e,sys)

    