from flask import Flask,request,render_template
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# upload the file and select the dependent feature
@app.route('/upload', methods=['GET','POST'])
def upload():
    features=None
    if request.method == 'POST':
        file=request.files['file']        
        dataingestion = DataIngestion()
        
        # we will send this file to data_ingestion which will split data into test and train then save it to_csv
        features = dataingestion.initiate_data_ingestion(file)
    
    return render_template('upload.html',features=features)

@app.route('/training',methods=['POST'])
def feature_selection():
    selected_feature = request.form.get('selected_feature')

    # we will first remove dependent feature from the training dataset
    # then we will do preprocessing
    # then make pickle file
    # then model selction and hyperparameter tuning
    # then making pickle file of best model
    
    return render_template('training.html', selected_feature=selected_feature)


if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)