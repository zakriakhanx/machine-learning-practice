"""
Stock Price Prediction - Next Day Closing Price
================================================
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_stock_data(ticker, period='1y'):
    """
    Load historical stock data using yfinance library.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL' for Apple)
        period: Data period to fetch (default: '1y' for 1 year)
    
    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Volume)
    """
    # Create a Ticker object for the given stock symbol
    stock = yf.Ticker(ticker)
    # Fetch historical market data for the specified period
    df = stock.history(period=period)
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

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
    # Copy the dataframe to avoid modifying the original
    df = df.copy()
    
    # Create target variable: next day's closing price
    # Shift Close back by 1 day so current features predict tomorrow
    # e.g., Monday's features predict Tuesday's close
    df['Next_Close'] = df['Close'].shift(-1)
    
    # Drop the last row which has NaN target (no next day available)
    df = df.dropna()
    
    # Define features: OHLCV columns used for prediction
    X = df[['Open', 'High', 'Low', 'Volume']]
    
    # Define target: Next day's closing price
    y = df['Next_Close']
    
    return X, y


# =============================================================================
# MODEL TRAINING
# =============================================================================

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
    # Select model based on type parameter
    if model_type == 'linear':
        model = LinearRegression()
    else:
        # Random Forest: ensemble of 100 decision trees
        # random_state=42 ensures reproducible results
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model by fitting it to the training data
    model.fit(X_train, y_train)
    return model


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_predictions(y_test, y_pred, title):
    """
    Plot actual vs predicted stock prices for visual comparison.
    
    Args:
        y_test: Actual stock prices (ground truth)
        y_pred: Predicted stock prices
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual values as a line
    plt.plot(y_test.values, label='Actual', alpha=0.7)
    
    # Plot predicted values as a line
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    
    # Add axis labels, title, legend, and grid for readability
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Main function to run the stock prediction pipeline.
    
    Steps:
    1. Load historical stock data using yfinance
    2. Prepare features (Open, High, Low, Volume) and target (Next Close)
    3. Split data into train (80%) and test (20%) sets (time series split)
    4. Train and evaluate Linear Regression model
    5. Train and evaluate Random Forest model
    6. Compare both models using R2 score
    """
    # Use Apple stock as the default ticker
    ticker = 'AAPL'
    
    print(f"Loading data for {ticker}...")
    
    # Step 1: Load 1 year of historical stock data
    df = load_stock_data(ticker)
    
    # Display data summary
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    # Step 2: Prepare features and target variable
    X, y = prepare_features(df)
    
    # Step 3: Split data preserving temporal order (no shuffling for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Step 4: Train and evaluate Linear Regression
    print("\n--- Linear Regression ---")
    lr_model = train_model(X_train, y_train, 'linear')
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)  # R2 measures goodness of fit
    print(f"R2 Score: {lr_r2:.4f}")
    plot_predictions(
        y_test, lr_pred, 
        f'{ticker} - Linear Regression: Actual vs Predicted Next-Day Close'
    )
    
    # Step 5: Train and evaluate Random Forest
    print("\n--- Random Forest ---")
    rf_model = train_model(X_train, y_train, 'random_forest')
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"R2 Score: {rf_r2:.4f}")
    plot_predictions(
        y_test, rf_pred, 
        f'{ticker} - Random Forest: Actual vs Predicted Next-Day Close'
    )
    
    # Step 6: Compare model performance
    print("\n--- Model Comparison ---")
    print(f"Linear Regression - R2: {lr_r2:.4f}")
    print(f"Random Forest - R2: {rf_r2:.4f}")
    
    # Report which model performed better
    if rf_r2 > lr_r2:
        print("\nRandom Forest performs better for this dataset.")
    else:
        print("\nLinear Regression performs better for this dataset.")


if __name__ == '__main__':
    main()