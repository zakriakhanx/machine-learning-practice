import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def prepare_features(df):
    df = df.copy()
    df['Next_Close'] = df['Close'].shift(-1)
    df = df.dropna()
    
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Next_Close']
    
    return X, y

def train_model(X_train, y_train, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    return model

def plot_predictions(y_test, y_pred, title):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    ticker = 'AAPL'
    print(f"Loading data for {ticker}...")
    
    df = load_stock_data(ticker)
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    X, y = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    print("\n--- Linear Regression ---")
    lr_model = train_model(X_train, y_train, 'linear')
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    print(f"R2 Score: {lr_r2:.4f}")
    plot_predictions(y_test, lr_pred, f'{ticker} - Linear Regression: Actual vs Predicted Next-Day Close')
    
    print("\n--- Random Forest ---")
    rf_model = train_model(X_train, y_train, 'random_forest')
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"R2 Score: {rf_r2:.4f}")
    plot_predictions(y_test, rf_pred, f'{ticker} - Random Forest: Actual vs Predicted Next-Day Close')
    
    print("\n--- Model Comparison ---")
    print(f"Linear Regression - R2: {lr_r2:.4f}")
    print(f"Random Forest - R2: {rf_r2:.4f}")
    
    if rf_r2 > lr_r2:
        print("\nRandom Forest performs better for this dataset.")
    else:
        print("\nLinear Regression performs better for this dataset.")

if __name__ == '__main__':
    main()