# Heart Disease Prediction

End-to-end heart disease prediction pipeline using the UCI Heart Disease dataset.

## Overview

This project implements a complete machine learning pipeline for predicting heart disease:

- Data loading and exploratory analysis
- Data cleaning and preprocessing
- Missing value imputation using KNN
- Logistic regression model training
- Model evaluation with ROC curves and confusion matrices
- SHAP-based feature importance analysis

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn
- shap

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn shap
```

## Usage

Run the script:

```bash
python predict.py
```

The script performs a complete pipeline:

1. Loads heart disease data from `heart.csv`
2. Exploratory Data Analysis (EDA)
3. Data quality assessment
4. Categorical encoding
5. Missing value imputation (KNN)
6. Model training (Logistic Regression)
7. Model evaluation (ROC curve, confusion matrix)
8. SHAP feature importance analysis

## Dataset

The UCI Heart Disease dataset contains clinical features for patients:

| Feature | Description |
|---------|-------------|
| Age | Patient age |
| Sex | Gender (M/F) |
| ChestPainType | Chest pain type (ATA, NAP, ASY, TA) |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | Fasting blood sugar |
| RestingECG | Resting ECG results |
| MaxHR | Maximum heart rate |
| ExerciseAngina | Exercise-induced angina |
| Oldpeak | ST depression |
| ST_Slope | ST segment slope |
| HeartDisease | Target (1=Disease, 0=Healthy) |

## Pipeline Steps

1. **Data Loading**: Load CSV data into pandas DataFrame
2. **EDA**: Shape, data types, descriptive statistics
3. **Data Quality**: Missing values, duplicates, cardinality
4. **Categorical Encoding**: Convert string columns to numeric
5. **KNN Imputation**: Handle missing zeros in Cholesterol and RestingBP
6. **Data Splitting**: 80/20 train/test split with stratification
7. **Model Training**: Logistic Regression
8. **Evaluation**: ROC curve, confusion matrix
9. **SHAP Analysis**: Feature importance visualization

## Project Structure

- `predict.py` - Main prediction pipeline
- `heart.csv` - Heart disease dataset
- `task.md` - Original task description

## Model Output

The script outputs:

- Logistic Regression accuracy
- ROC curve plot
- Confusion matrix visualization
- SHAP summary plot showing feature importance
