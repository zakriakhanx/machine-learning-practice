"""
House Price Prediction Model
============================
This script builds a Linear Regression model to predict house prices using the Housing.csv dataset.
It includes data preprocessing, model training, evaluation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# =============================================================================
# STEP 1: Data Loading and Exploration
# =============================================================================
# Load and inspect the housing dataset to understand its structure and quality.

# Load the housing dataset from CSV file
housing_data = pd.read_csv('./Housing.csv')

# Display the first few rows to understand the data structure
housing_data.head()

# Show data types, missing values, and memory usage
housing_data.info()

# Generate descriptive statistics for all columns (numeric and categorical)
housing_data.describe(include='all')

# Count missing values in each column
housing_data.isnull().sum()

# Count duplicate rows in the dataset
housing_data.duplicated().sum()

# Count unique values in each column
housing_data.nunique()

# =============================================================================
# STEP 2: Data Preprocessing
# =============================================================================
# Transform categorical variables into numeric format suitable for regression.

# Convert categorical variables to dummy/indicator variables (one-hot encoding)
# drop_first=True removes multicollinearity by dropping the first category
# dtype=int converts boolean columns to integers (0/1) for numerical stability
housing_data = pd.get_dummies(housing_data, drop_first=True, dtype=int)

# Display the transformed data
print(housing_data.head())

# =============================================================================
# STEP 3: Feature and Target Separation
# =============================================================================
# Separate the dataset into features (X) and target variable (y).

# Features (X): All columns except the target variable 'price'
X = housing_data.drop('price', axis=1)

# Target (y): The house price we want to predict
y = housing_data['price']

# =============================================================================
# STEP 4: Train-Test Split
# =============================================================================
# Split data into training (80%) and testing (20%) sets for model validation.

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,     # 20% of data reserved for testing
    random_state=42   # Seed for reproducible results
)

# =============================================================================
# STEP 5: Feature Scaling
# =============================================================================
# Standardize features to have zero mean and unit variance.
# This ensures all features contribute equally and helps gradient descent converge faster.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test_scaled = scaler.transform(X_test)        # Only transform test data (no data leakage)

# =============================================================================
# STEP 6: Model Training
# =============================================================================
# Train a Linear Regression model to learn the relationship between features and price.

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# =============================================================================
# STEP 7: Prediction
# =============================================================================
# Generate predictions on the test set using the trained model.

y_pred = model.predict(X_test_scaled)

# =============================================================================
# STEP 8: Model Evaluation
# =============================================================================
# Evaluate model performance using standard regression metrics.

# Mean Absolute Error (MAE): Average absolute difference between actual and predicted
# Interpretable in the same units as the target variable (e.g., dollars)
mae = mean_absolute_error(y_test, y_pred)

# Root Mean Squared Error (RMSE): Square root of average squared errors
# Penalizes larger errors more heavily than MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# =============================================================================
# STEP 9: Visualization
# =============================================================================
# Plot actual vs predicted prices to visually assess model performance.

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')

# Add diagonal reference line (perfect prediction line where actual = predicted)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# Label axes and add title
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')

# Adjust layout to prevent label clipping
plt.tight_layout()
plt.show()