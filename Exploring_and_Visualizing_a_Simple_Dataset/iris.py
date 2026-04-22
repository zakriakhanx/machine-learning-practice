"""
Iris Dataset Exploration and Visualization
============================================
This script loads the Iris dataset and performs exploratory data analysis (EDA)
through various statistical summaries and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# STEP 1: Load the Data
# =============================================================================
# Read the Iris dataset from a CSV file into a pandas DataFrame.
# The Iris dataset contains measurements for 150 flower samples across 3 species,
# with features: sepal length, sepal width, petal length, and petal width (in cm).
iris = pd.read_csv('iris.csv')

# =============================================================================
# STEP 2: Data Exploration - Statistical Summaries
# =============================================================================
# Print the dimensions of the dataset (rows, columns)
print(iris.shape)

# Print the column names to identify all features
print(iris.columns)

# Display the first 5 rows for an initial preview of the data
print(iris.head())

# Display data types, missing values, and memory usage
print(iris.info())

# Generate descriptive statistics (count, mean, std, min, quartiles, max)
# This helps understand the distribution and scale of each numeric feature
print(iris.describe())

# =============================================================================
# STEP 3: Scatter Plots
# =============================================================================
# Scatter plots visualize relationships between two numeric features,
# colored by species to reveal how species cluster differently.

# Plot sepal dimensions (length vs width)
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Plot petal dimensions (length vs width)
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=iris)
plt.title('Petal Length vs Petal Width')
plt.show()

# =============================================================================
# STEP 4: Histograms
# =============================================================================
# Histograms show the distribution of a single feature, with KDE (kernel density
# estimate) overlaid to visualize the probability density curve.

# Distribution of petal length across species
sns.histplot(data=iris, x='PetalLengthCm', hue='Species', kde=True)
plt.title('Distribution of Petal Length by Species')
plt.show()

# Distribution of sepal length across species
sns.histplot(data=iris, x='SepalLengthCm', hue='Species', kde=True)
plt.title('Distribution of Sepal Length by Species')
plt.show()

# =============================================================================
# STEP 5: Box Plots
# =============================================================================
# Box plots display the median, quartiles, and potential outliers for each
# species, making it easy to compare distributions across categories.

# Box plot of petal width by species
sns.boxplot(x='Species', y='PetalWidthCm', data=iris)
plt.title('Boxplot of Petal Width by Species')
plt.show()

# Box plot of sepal width by species
sns.boxplot(x='Species', y='SepalWidthCm', data=iris)
plt.title('Boxplot of Sepal Width by Species')
plt.show()