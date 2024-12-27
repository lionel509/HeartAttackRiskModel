# Heart Attack Prediction Model

This repository contains a machine learning project for predicting heart attack risk using patient characteristics. The project compares multiple ML models and implements the best performing one as a neural network.

## Project Structure

- `model_design/` - Model evaluation and comparison
  - `ML_model_comparison.ipynb` - Notebook comparing different ML algorithms
  - `results/` - Performance metrics and visualizations for each model
  - `heart.csv` - Training dataset

- `Prediction/` - Final model implementation  
  - `main.ipynb` - Neural network implementation and training
  - `prediction.py` - Prediction script
  - `heart_attack_model_v1.h5` - Trained model weights
  - `heart_attack_model_v2.h5` - Improved model weights

## Dataset

The model uses the [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) from Kaggle. Features include:

- Age
- Sex 
- Chest pain type
- Blood pressure
- Cholesterol
- And other medical indicators

## Models Evaluated

- Logistic Regression
- Random Forest
- Gradient Boosting
- AdaBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors
- Decision Tree
- Naive Bayes
- Linear Discriminant Analysis (LDA)
- XGBoost

## Results

The models achieved the following performances (from `results/` directory):

- K-Nearest Neighbors: 91.8% accuracy
- SVM: 86.9% ROC AUC
- LDA: 93.5% ROC AUC 
- Random Forest: 92.4% ROC AUC

The final neural network implementation achieved 84% test accuracy.

## Requirements

Required Python packages are listed in `requirements.txt`:

```sh
pandas
numpy 
scikit-learn
xgboost
matplotlib
seaborn
tensorflow
keras-tuner

# useage 

## 1. install dependicites 
```bash 
pip install -r requirements.txt

## 2. Run comparsion model
```bash 
jupyter notebook model_design/ML_model_comparison.ipynb

## or Train neural network 

```bash 
jupyter notebook Prediction/main.ipynb