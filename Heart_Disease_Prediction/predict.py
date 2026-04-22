"""
Heart Disease Prediction Pipeline
=================================
This script performs end-to-end heart disease prediction using the UCI Heart Disease dataset.
It includes data loading, exploratory analysis, preprocessing, model training, and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import shap

# =============================================================================
# STEP 1: Data Loading
# =============================================================================
# Load the heart disease dataset from a CSV file.
# The dataset contains clinical features for patients including demographics,
# symptoms, and test results to predict heart disease presence.
heart_df = pd.read_csv('./heart.csv')

# =============================================================================
# STEP 2: Exploratory Data Analysis (EDA)
# =============================================================================
# Explore the dataset to understand its structure, content, and characteristics.

# Display the complete dataset (all rows and columns)
heart_df

# Display a random sample of 5 rows for initial inspection
heart_df.sample(5)

# Display dataset metadata: row count, column names, data types, non-null counts
heart_df.info()

# Generate descriptive statistics for all numeric columns (mean, std, min, max, quartiles)
heart_df.describe()

# Generate comprehensive statistics including both numeric and categorical columns
heart_df.describe(include='all')

# =============================================================================
# STEP 3: Data Quality Assessment
# =============================================================================
# Identify data quality issues such as missing values, duplicates, and cardinality.

# Count missing (null) values in each column
heart_df.isnull().sum()

# Count duplicate records to check for data redundancy
heart_df.duplicated().sum()

# Count unique values per column to understand feature cardinality
heart_df.nunique()

# =============================================================================
# STEP 4: Categorical Encoding
# =============================================================================
# Convert categorical (string) variables to numeric codes for machine learning.

# Identify columns with string (object) data type
cat_col = heart_df.select_dtypes(include=['str']).columns
cat_col

# Apply label encoding to each categorical column
# Encoding scheme maps each unique category to an integer:
#   Sex: M=0, F=1
#   ChestPainType: ATA=0, NAP=1, ASY=2, TA=3
#   RestingECG: Normal=0, ST=1, LVH=2
#   ExerciseAngina: N=0, Y=1
#   ST_Slope: Up=0, Flat=1, Down=2
for col in cat_col:
    print(col)
    # Show mapping of unique values to their encoded integers
    print((heart_df[col].unique()), list(range(heart_df[col].nunique())))
    
    # Replace original categorical values with sequential integers [0, 1, 2, ...]
    heart_df[col] = heart_df[col].replace(heart_df[col].unique(), list(range(heart_df[col].nunique())))
    print('*' * 90)
    print()

# Display the transformed dataset after encoding
heart_df

# =============================================================================
# STEP 5: Missing Value Imputation
# =============================================================================
# Handle biologically impossible zero values (missing data) using KNN imputation.

# -----------------------------------------------------------------------------
# 5a: Cholesterol Imputation
# -----------------------------------------------------------------------------
# Cholesterol value of 0 mg/dl is biologically impossible, treat as missing.
heart_df['Cholesterol'].value_counts()

# Replace all zero values with NaN (Not a Number) to mark as missing
heart_df['Cholesterol'] = heart_df['Cholesterol'].replace(0, np.nan)

# Initialize K-Nearest Neighbors imputer with k=3 neighbors
# KNN imputation estimates missing values based on the 3 most similar patients
imputer = KNNImputer(n_neighbors=3)

# Fit the imputer on the data and transform it (impute missing values)
after_impute = imputer.fit_transform(heart_df)
heart_df = pd.DataFrame(after_impute, columns=heart_df.columns)

# Verify imputation was successful (should return 0 missing values)
heart_df['Cholesterol'].isna().sum()

# -----------------------------------------------------------------------------
# 5b: Resting Blood Pressure Imputation
# -----------------------------------------------------------------------------
# Resting BP of 0 mm Hg is physiologically impossible, treat as missing.
heart_df['RestingBP'][heart_df['RestingBP'] == 0]

# Replace zero values with NaN
heart_df['RestingBP'] = heart_df['RestingBP'].replace(0, np.nan)

# Initialize a second KNN imputer for blood pressure
imputer2 = KNNImputer(n_neighbors=3)

# Apply imputation and reconstruct DataFrame
after_impute2 = imputer2.fit_transform(heart_df)
heart_df = pd.DataFrame(after_impute2, columns=heart_df.columns)

# Verify imputation was successful
heart_df['RestingBP'].isnull().sum()

# =============================================================================
# STEP 6: Data Type Optimization
# =============================================================================
# Convert columns to appropriate data types for efficiency and correctness.

# After KNN imputation, all columns become float64 by default
# Convert discrete variables back to integers for memory efficiency

# Get all column names
without_oldPeak = heart_df.columns

# Exclude 'Oldpeak' column as it contains continuous decimal values (ST depression)
without_oldPeak = without_oldPeak.drop('Oldpeak')

# Convert all columns (except Oldpeak) to 32-bit integers
heart_df[without_oldPeak] = heart_df[without_oldPeak].astype('int32')

# Display final dataset metadata to confirm type conversion
heart_df.info()

# =============================================================================
# STEP 7: Data Splitting
# =============================================================================
# Split the dataset into training and testing sets for model evaluation.

# Split 80% training / 20% testing with stratified sampling to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    heart_df.drop('HeartDisease', axis=1),  # Features: all columns except target
    heart_df['HeartDisease'],                # Target: binary indicator (0=No Disease, 1=Disease)
    test_size=0.2,                           # Reserve 20% of data for testing
    random_state=42,                         # Seed for reproducible results
    stratify=heart_df['HeartDisease']        # Maintain class distribution in train/test sets
)

# =============================================================================
# STEP 8: Logistic Regression Model Training
# =============================================================================
# Train a logistic regression classifier to predict heart disease.

# Logistic Regression is a linear model that outputs probability of binary outcome
# Uses sigmoid function to map linear combinations of features to [0, 1] range

# Initialize and train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Generate predictions on the test set
log_reg_pred = log_reg.predict(X_test)

# Calculate accuracy score and convert to percentage
lr_acc = accuracy_score(y_test, log_reg_pred) * 100

# Display the model's accuracy
print(f"LR Accuracy: {lr_acc:.2f}%")

# =============================================================================
# STEP 9: Model Evaluation - ROC Curve
# =============================================================================
# Plot the Receiver Operating Characteristic (ROC) curve to visualize
# the trade-off between true positive rate and false positive rate.

# Calculate ROC curve data points
fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])

# Plot the ROC curve
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# =============================================================================
# STEP 10: Model Evaluation - Confusion Matrix
# =============================================================================
# Display a confusion matrix to show classification results across both classes.

# Generate the confusion matrix comparing true labels vs predicted labels
cm = confusion_matrix(y_test, log_reg_pred)

# Visualize the confusion matrix with labeled display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Heart Disease'])
disp.plot(cmap='Blues')
plt.show()

# =============================================================================
# STEP 11: Feature Importance Analysis with SHAP
# =============================================================================
# Use SHAP (SHapley Additive exPlanations) to explain model predictions
# and identify which features have the most impact on heart disease prediction.

# Create a SHAP explainer for the logistic regression model
explainer = shap.Explainer(log_reg, X_train)

# Calculate SHAP values for all test samples
shap_values = explainer(X_test)

# Generate a summary plot showing feature importance and effect direction
shap.summary_plot(shap_values, X_test)