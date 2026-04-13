"""
Stock Price Prediction - Next Day Closing Price
=================================================
This script uses historical stock data to predict the next day's closing price
using Linear Regression and Random Forest models.

Features used: Open, High, Low, Volume
Target: Next day's Close price
"""

import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_stock_data(ticker, period='1y'):
    """
    Load historical stock data using yfinance library.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL' for Apple)
        period: Data period to fetch (default: '1y' for 1 year)
    
    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Volume)
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df


def prepare_features(df):
    """
    Prepare features and target variable for model training.
    
    Creates a target variable (Next_Close) by shifting the Close price
    back by 1 day, so each row's features predict the next day's close.
    
    Args:
        df: DataFrame with stock data
    
    Returns:
        X: Feature matrix (Open, High, Low, Volume)
        y: Target vector (Next day's Close price)
    """
    df = df.copy()
    
    # Create target variable: next day's closing price
    # Shift Close back by 1 day so current features predict tomorrow
    df['Next_Close'] = df['Close'].shift(-1)
    
    # Drop the last row which has no next day (NaN)
    df = df.dropna()
    
    # Features: Open, High, Low, Volume
    X = df[['Open', 'High', 'Low', 'Volume']]
    
    # Target: Next day's closing price
    y = df['Next_Close']
    
    return X, y


def train_model(X_train, y_train, model_type='linear'):
    """
    Train a regression model on the provided data.
    
    Args:
        X_train: Training features
        y_train: Training target values
        model_type: 'linear' for LinearRegression, anything else for RandomForest
    
    Returns:
        Trained model object
    """
    if model_type == 'linear':
        model = LinearRegression()
    else:
        # Random Forest with 100 trees, fixed random state for reproducibility
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    return model


def plot_predictions(y_test, y_pred, title):
    """
    Plot actual vs predicted stock prices for visual comparison.
    
    Args:
        y_test: Actual stock prices (ground truth)
        y_pred: Predicted stock prices
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(y_test.values, label='Actual', alpha=0.7)
    
    # Plot predicted values
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    
    # Add labels and formatting
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the stock prediction pipeline.
    
    Steps:
    1. Load historical stock data
    2. Prepare features and target
    3. Split data into train/test sets
    4. Train and evaluate Linear Regression
    5. Train and evaluate Random Forest
    6. Compare and report results
    """
    # Use Apple stock as the default ticker
    ticker = 'AAPL'
    
    print(f"Loading data for {ticker}...")
    
    # Load 1 year of historical data
    df = load_stock_data(ticker)
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    # Prepare features and target
    X, y = prepare_features(df)
    
    # Split data: 80% training, 20% testing
    # shuffle=False preserves time order for time series data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train and evaluate Linear Regression
    print("\n--- Linear Regression ---")
    lr_model = train_model(X_train, y_train, 'linear')
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    print(f"R2 Score: {lr_r2:.4f}")
    plot_predictions(
        y_test, lr_pred, 
        f'{ticker} - Linear Regression: Actual vs Predicted Next-Day Close'
    )
    
    # Train and evaluate Random Forest
    print("\n--- Random Forest ---")
    rf_model = train_model(X_train, y_train, 'random_forest')
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"R2 Score: {rf_r2:.4f}")
    plot_predictions(
        y_test, rf_pred, 
        f'{ticker} - Random Forest: Actual vs Predicted Next-Day Close'
    )
    
    # Compare model performance
    print("\n--- Model Comparison ---")
    print(f"Linear Regression - R2: {lr_r2:.4f}")
    print(f"Random Forest - R2: {rf_r2:.4f}")
    
    if rf_r2 > lr_r2:
        print("\nRandom Forest performs better for this dataset.")
    else:
        print("\nLinear Regression performs better for this dataset.")


if __name__ == '__main__':
    main()