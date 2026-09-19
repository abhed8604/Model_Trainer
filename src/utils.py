import os
import sys
import dill

from sklearn.metrics import r2_score # we are using r2 score here but can also use RMS or MAE here
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException
from src.logger import logging

def save_pickle(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]

            logging.info(f'testing {list(models.keys())[i]} for hyperparameter tuning')

            gs = GridSearchCV(
                # doing grid search for hyperparameter tuning
                estimator=model,
                param_grid=para,
                n_jobs=-1, # we use n_jobs to use parallel processing
                cv=3
            )
            gs.fit(X_train,y_train)

            best_model = gs.best_estimator_
            del gs # we delete the model to free up memory
            
            y_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_pred)

            logging.info(f'{list(models.keys())[i]} gave R2 score of {test_score}')

            report[list(models.keys())[i]]={
                "best_model": best_model,
                "score": test_score
            }

        return report
                 
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path): # we will use this function to load the model from pickle file
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)