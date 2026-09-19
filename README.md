# Model Trainer

A simple end-to-end machine learning web app built with Flask to automate the tedious parts of building regression pipelines. 

You just upload a CSV, pick which column you want to predict, and it takes care of splitting the data, preprocessing (handling nulls, one-hot encoding, feature scaling), training multiple regressors with GridSearchCV hyperparameter tuning, and saving the best model. After training, you get a clean prediction form to test inputs in real time.

## How it works

1. **Ingest & Split**: Takes your CSV file and splits it into 75% train and 25% test sets.
2. **Preprocessing**: Automatically separates numerical and categorical columns, imputes missing values, scales features, one-hot encodes categories, and serializes the pipeline objects.
3. **Training & Tuning**: Runs GridSearchCV across multiple models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost, AdaBoost, KNN) to find the best estimator by \(R^2\) score.
4. **Prediction**: Inspects the saved preprocessing metadata to build a dynamic form (dropdowns for categories, numeric inputs for floats/ints) and runs predictions using the top model.

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/abhed8604/Model_Trainer.git
cd Model_Trainer
```

### 2. Environment Setup
I create and activate the Conda environment directly inside the project directory:
```bash
conda create -p ./venv python=3.8 -y
conda activate ./venv
```
*(Or use standard venv if you prefer: `python -m venv venv && source venv/bin/activate`)*

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

### Alternative: Run with Docker

1. Build the image:
```bash
docker build -t model-trainer .
```

2. Run the container:
```bash
docker run -p 5000:5000 model-trainer
```
Open `http://localhost:5000` in your browser.

## Project Layout

- `app.py`: Flask routes (`/`, `/upload`, `/training`, `/api/train`, `/predict`)
- `Dockerfile`: Container configuration for packaging and running the application
- `src/components/data_ingestion.py`: CSV ingestion and train/test splitting
- `src/components/data_transformation.py`: Pipelines for numerical & categorical features
- `src/components/model_training.py`: Grid search hyperparameter tuning & model selection
- `src/components/predict.py`: Inference pipeline loading saved artifacts
- `templates/`: Lightweight, responsive UI templates for upload, training progress, and inference
