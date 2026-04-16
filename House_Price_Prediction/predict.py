import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

housing_data = pd.read_csv('./Housing.csv')
housing_data.head()
housing_data.info()
housing_data.describe(include='all')
housing_data.isnull().sum()
housing_data.duplicated().sum()
housing_data.nunique()


print(list(housing_data.select_dtypes(include='str').columns))

housing_data['Location'].unique()
housing_data['Location'] = housing_data['Location'].replace(housing_data['Location'].unique(), list(range(housing_data['Location'].nunique())))
housing_data['Location'] = housing_data['Location'].astype('int64')
# ['Downtown', 'Suburban', 'Urban', 'Rural']
# [   0,           1,         2,      3]

housing_data['Condition'].unique()
housing_data['Condition'] = housing_data['Condition'].replace(housing_data['Condition'].unique(), list(range(housing_data['Condition'].nunique()))[::-1])
housing_data['Condition'] = housing_data['Condition'].astype('int64')
# ['Excellent', 'Good', 'Fair', 'Poor']
# [     3,        2,      1,      0]

housing_data['Garage'].unique()
housing_data['Garage'] = housing_data['Garage'].replace(housing_data['Garage'].unique(), list(range(housing_data['Garage'].nunique())))
housing_data['Garage'] = housing_data['Garage'].astype('int64')
# ['No', 'Yes']
# [ 0,     1]

def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X_train, X_test, y_train, y_test = train_test_split(
    housing_data.drop('Price', axis=1),
    housing_data['Price'],
    test_size=0.2,
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.show()