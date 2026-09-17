import os
import pandas as pd

from flask import Flask,request,render_template,jsonify

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.utils import load_object
from src.components.predict import PredictPipeline

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
def training_page():    
    selected_feature = request.form.get('selected_feature')
    
    return render_template('training.html',selected_feature=selected_feature)

@app.route('/api/train', methods=['POST'])
def run_training():
    data = request.get_json()
    selected_feature = data.get('selected_feature')
    
    datatransformation=DataTransformation()
    train_arr,test_arr=datatransformation.initiate_preprocessing(selected_feature)
    
    modeltrainer=ModelTrainer()
    best_model_name,score,model_report=modeltrainer.initiate_model_trainer(train_arr,test_arr)
    
    return jsonify({
        "status": "success",
        "best_model_name": best_model_name,
        "score": round(score, 4),
        "model_report": {k: {"score": round(v["score"], 4)} for k, v in model_report.items()}
    })


@app.route('/predict', methods=['GET','POST'])
def predictor():
    preprocessor_path=os.path.join('artifacts', 'preprocessor.pkl')
    cat_info_path=os.path.join('artifacts', 'cat_feature_info.pkl')

    preprocessor=load_object(preprocessor_path)
    features = list(preprocessor.feature_names_in_) # we use this to get all the feature name as we cant get our feature list from upload route
    cat_feature_info = load_object(cat_info_path)
    
    if request.method == 'GET':
        return render_template('data_prediction.html',features=features,cat_feature_info=cat_feature_info)

    # POST after form
    data = {feature: [request.form.get(feature)] for feature in features}
    pred_df = pd.DataFrame(data)

    for col in pred_df.columns:
        if col not in cat_feature_info:
            pred_df[col] = pd.to_numeric(pred_df[col])

    pipeline = PredictPipeline()
    preds = pipeline.predict(pred_df)
    result = round(preds[0], 2) # we are rounding the result to make it look clean
    
    return render_template(
        'data_prediction.html',
        features=features,
        cat_feature_info=cat_feature_info,
        result=result
    )

if __name__=='__main__':
    print("\nServer running! Open: http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0',port=5000)