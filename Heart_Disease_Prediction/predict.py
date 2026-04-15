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

# ========================= DATA LOADING =========================

# Load raw heart disease dataset
heart_df = pd.read_csv('./heart.csv')

# ========================= EXPLORATORY DATA ANALYSIS =========================

# Display complete dataset
heart_df

# Display random sample of 5 rows for initial inspection
heart_df.sample(5)

# Display dataset structure: row count, column types, non-null counts, memory usage
heart_df.info()

# Generate statistical summary for numerical columns (mean, std, min, max, quartiles)
heart_df.describe()

# Generate comprehensive summary including categorical columns
heart_df.describe(include='all')

# ========================= DATA QUALITY ASSESSMENT =========================

# Count missing (null) values per column
heart_df.isnull().sum()

# Identify duplicate records to prevent data redundancy
heart_df.duplicated().sum()

# Count unique values per column to understand cardinality
heart_df.nunique()

# ========================= CATEGORICAL ENCODING =========================

# Identify columns with object (string) data type
cat_col = heart_df.select_dtypes(include=['str']).columns
cat_col

# Convert categorical variables to numeric using label encoding
# Encoding scheme:
#   Sex: M=0, F=1
#   ChestPainType: ATA=0, NAP=1, ASY=2, TA=3
#   RestingECG: Normal=0, ST=1, LVH=2
#   ExerciseAngina: N=0, Y=1
#   ST_Slope: Up=0, Flat=1, Down=2
for col in cat_col:
    print(col)
    # Display unique values and their assigned numeric codes
    print((heart_df[col].unique()), list(range(heart_df[col].nunique())))
    
    # Replace categorical values with sequential integers
    heart_df[col] = heart_df[col].replace(heart_df[col].unique(), list(range(heart_df[col].nunique())))
    print('*' * 90)
    print()

# Display transformed dataset
heart_df

# ========================= MISSING VALUE IMPUTATION =========================

# -------------------- Cholesterol Imputation --------------------
# Identify biologically impossible cholesterol values (0 mg/dl)
heart_df['Cholesterol'].value_counts()

# Replace zero values with NaN to mark as missing
heart_df['Cholesterol'] = heart_df['Cholesterol'].replace(0, np.nan)

# Initialize K-Nearest Neighbors imputer (k=3)
# KNN imputation estimates missing values based on 3 most similar patients
imputer = KNNImputer(n_neighbors=3)

# Apply imputation and reconstruct DataFrame
after_impute = imputer.fit_transform(heart_df)
heart_df = pd.DataFrame(after_impute, columns=heart_df.columns)

# Verify successful imputation (should return 0)
heart_df['Cholesterol'].isna().sum()

# -------------------- Resting Blood Pressure Imputation --------------------
# Identify biologically impossible resting BP values (0 mm Hg)
heart_df['RestingBP'][heart_df['RestingBP'] == 0]

# Replace zero values with NaN
heart_df['RestingBP'] = heart_df['RestingBP'].replace(0, np.nan)

# Initialize second KNN imputer for blood pressure
imputer2 = KNNImputer(n_neighbors=3)

# Apply imputation and reconstruct DataFrame
after_impute2 = imputer2.fit_transform(heart_df)
heart_df = pd.DataFrame(after_impute2, columns=heart_df.columns)

# Verify successful imputation
heart_df['RestingBP'].isnull().sum()

# ========================= DATA TYPE OPTIMIZATION =========================

# After KNN imputation, all columns become float64
# Convert discrete variables back to integers for memory efficiency and semantic correctness

# Get all column names
without_oldPeak = heart_df.columns

# Exclude 'Oldpeak' column as it requires continuous decimal values
without_oldPeak = without_oldPeak.drop('Oldpeak')

# Convert all columns except Oldpeak to 32-bit integers
heart_df[without_oldPeak] = heart_df[without_oldPeak].astype('int32')

# Display final dataset structure
heart_df.info()

# ========================= DATA SPLITTING =========================

# Split dataset into training (80%) and testing (20%) sets
# Features (X): All clinical variables except target
# Target (y): Binary heart disease indicator (0 = No Disease, 1 = Disease)
X_train, X_test, y_train, y_test = train_test_split(
    heart_df.drop('HeartDisease', axis=1),  # Feature matrix (all columns except target)
    heart_df['HeartDisease'],                # Target variable (binary classification)
    test_size=0.2,                           # Reserve 20% of data for testing
    random_state=42,                         # Seed for reproducible splits
    stratify=heart_df['HeartDisease']        # Maintain class distribution in both sets
)

# ========================= LOGISTIC REGRESSION TRAINING =========================

# Logistic Regression: Linear model that estimates probability of binary outcomes
# Uses sigmoid function to map linear combination of features to [0,1] probability range

# Train final logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Generate predictions on test set
log_reg_pred = log_reg.predict(X_test)

# Calculate and convert accuracy to percentage
lr_acc = accuracy_score(y_test, log_reg_pred) * 100

# Display accuracy of logistic regression model
print(f"LR Accuracy: {lr_acc:.2f}%")

# Display ROC curve for logistic regression model
fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Generate the matrix
cm = confusion_matrix(y_test, log_reg_pred)

# Visualize it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Heart Disease'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Feature importance analysis using SHAP values
explainer = shap.Explainer(log_reg, X_train)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)