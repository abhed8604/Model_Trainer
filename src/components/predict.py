import sys
import os
import pandas as pd

from src.utils import load_object
from src.exceptions import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,pred_data):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_scaled=preprocessor.transform(pred_data)
            preds=model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e,sys)