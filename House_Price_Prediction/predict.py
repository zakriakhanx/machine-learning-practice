"""
House Price Prediction Model
=============================
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
# DATA LOADING AND EXPLORATION
# =============================================================================

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
# DATA PREPROCESSING
# =============================================================================

# Convert categorical variables to dummy/indicator variables
# drop_first=True avoids multicollinearity by removing the first category
# dtype=int converts boolean columns to integers (0/1)
housing_data = pd.get_dummies(housing_data, drop_first=True, dtype=int)

# Display the transformed data
print(housing_data.head())

# =============================================================================
# FEATURE AND TARGET SEPARATION
# =============================================================================

# Separate features (X) from the target variable (price)
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

# Split data into training (80%) and testing (20%) sets
# random_state=42 ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,
    random_state=42
)

# =============================================================================
# FEATURE SCALING
# =============================================================================

# Standardize features by removing the mean and scaling to unit variance
# This helps linear regression converge faster and perform better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on training data
X_test_scaled = scaler.transform(X_test)        # Apply to test data

# =============================================================================
# MODEL TRAINING
# =============================================================================

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# =============================================================================
# PREDICTION
# =============================================================================

# Generate predictions on the test set
y_pred = model.predict(X_test_scaled)

# =============================================================================
# MODEL EVALUATION
# =============================================================================

# Calculate Mean Absolute Error (MAE) - average absolute difference between actual and predicted
mae = mean_absolute_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE) - penalizes larger errors more heavily
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# =============================================================================
# VISUALIZATION
# =============================================================================

# Create scatter plot comparing actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')

# Add a diagonal reference line (perfect prediction line)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# Label axes and add title
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')

# Adjust layout to prevent label clipping
plt.tight_layout()
plt.show()