# House Price Prediction

Linear Regression model for predicting house prices using the Housing dataset.

## Overview

This project implements a complete machine learning pipeline for predicting house prices:
- Data loading and exploration
- One-hot encoding for categorical variables
- Feature scaling with StandardScaler
- Linear Regression model training
- Model evaluation with MAE and RMSE
- Visualization of actual vs predicted prices

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

Run the script:
```bash
python predict.py
```

The script performs:
1. Loads housing data from `Housing.csv`
2. Data exploration and quality assessment
3. One-hot encoding of categorical variables
4. Train-test split (80/20)
5. Feature scaling
6. Linear Regression model training
7. Model evaluation (MAE, RMSE)
8. Scatter plot visualization

## Dataset

The Housing dataset contains features about houses:

| Feature | Description |
|---------|-------------|
| price | Target variable (house price) |
| area | Area of the house |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of stories |
| mainroad | Connected to main road (yes/no) |
| guestroom | Has guest room (yes/no) |
| basement | Has basement (yes/no) |
| hotwaterheating | Has hot water heating (yes/no) |
| airconditioning | Has air conditioning (yes/no) |
| parking | Number of parking spaces |
| prefarea | Preferred area (yes/no) |

## Pipeline Steps

1. **Data Loading**: Load CSV, explore with head(), info(), describe()
2. **Data Quality**: Check missing values, duplicates, unique counts
3. **Preprocessing**: One-hot encoding with drop_first=True
4. **Feature/Target Separation**: X = features, y = price
5. **Train-Test Split**: 80% train, 20% test
6. **Feature Scaling**: StandardScaler
7. **Model Training**: LinearRegression
8. **Evaluation**: MAE, RMSE metrics
9. **Visualization**: Actual vs Predicted scatter plot

## Project Structure

- `predict.py` - Main prediction pipeline
- `Housing.csv` - Housing dataset
- `task.md` - Original task description

## Model Output

The script outputs:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Scatter plot comparing actual vs predicted prices