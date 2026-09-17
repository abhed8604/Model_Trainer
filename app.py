from flask import Flask,request,render_template
import pandas as pd

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file=request.files['file']
    df = pd.read_csv(file)

    return render_template('upload.html')