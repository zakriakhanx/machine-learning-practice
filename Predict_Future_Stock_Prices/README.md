# Stock Price Prediction

Predict next day's stock closing price using Linear Regression and Random Forest models.

## Overview

This project builds machine learning models to predict next day's stock closing price:
- Downloads historical stock data using yfinance
- Uses OHLCV features (Open, High, Low, Volume)
- Trains and compares Linear Regression and Random Forest models
- Evaluates using R2 score
- Visualizes actual vs predicted prices

## Requirements

- Python 3.8+
- yfinance
- scikit-learn
- matplotlib

Install dependencies:
```bash
pip install yfinance scikit-learn matplotlib
```

## Usage

Run the script:
```bash
python predict.py
```

The script:
1. Loads 1 year of AAPL stock data via yfinance
2. Prepares features (Open, High, Low, Volume) and target (Next Close)
3. Splits data 80/20 (preserving temporal order)
4. Trains Linear Regression model
5. Trains Random Forest model
6. Compares R2 scores and displays visualizations

To use a different stock, modify the `ticker` variable in the `main()` function.

## Models

| Model | Description |
|-------|-------------|
| Linear Regression | Linear approach finding best fit line |
| Random Forest | Ensemble of 100 decision trees |

## Features

| Feature | Description |
|---------|-------------|
| Open | Opening price |
| High | Daily high price |
| Low | Daily low price |
| Volume | Trading volume |

**Target**: Next day's Close price

## Pipeline Steps

1. **Data Loading**: Fetch stock data via yfinance
2. **Feature Engineering**: Create Next_Close target via shift(-1)
3. **Train-Test Split**: 80/20 time series split (no shuffle)
4. **Linear Regression Training**: Fit and predict
5. **Random Forest Training**: Fit and predict
6. **Evaluation**: R2 score comparison
7. **Visualization**: Actual vs Predicted line plots

## Project Structure

- `predict.py` - Main prediction pipeline
- `task.md` - Original task description

## Model Output

- R2 Score for each model
- Line plots comparing actual vs predicted prices
- Model comparison summary