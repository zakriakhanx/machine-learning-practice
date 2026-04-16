import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

housing_data = pd.read_csv('./Housing.csv')
housing_data.head()
housing_data.info()
housing_data.describe(include='all')
housing_data.isnull().sum()
housing_data.duplicated().sum()
housing_data.nunique()