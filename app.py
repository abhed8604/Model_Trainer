from flask import Flask,request,render_template

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

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

    datatransformation=DataTransformation()
    train_arr,test_arr,cat_feature_info,a=datatransformation.initiate_preprocessing(selected_feature)
    
    modeltrainer=ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr,test_arr)
    
    return render_template('training.html', selected_feature=selected_feature)


@app.route('/predict', methods=['GET','POST'])
def predictor():
    pass

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)