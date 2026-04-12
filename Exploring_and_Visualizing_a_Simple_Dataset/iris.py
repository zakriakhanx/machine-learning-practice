import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv('iris.csv')

print(iris.shape)
print(iris.columns)
print(iris.head())
print(iris.info())
print(iris.describe())

#Scatter plot
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)
plt.title('Sepal Length vs Sepal Width')
plt.show()

sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=iris)
plt.title('Petal Length vs Petal Width')
plt.show()

# Histograms
sns.histplot(data=iris, x='PetalLengthCm', hue='Species', kde=True)
plt.title('Distribution of Petal Length by Species')
plt.show()

sns.histplot(data=iris, x='SepalLengthCm', hue='Species', kde=True)
plt.title('Distribution of Sepal Length by Species')
plt.show()

# Boxplots
sns.boxplot(x='Species', y='PetalWidthCm', data=iris)
plt.title('Boxplot of Petal Width by Species')
plt.show()

sns.boxplot(x='Species', y='SepalWidthCm', data=iris)
plt.title('Boxplot of Sepal Width by Species')
plt.show()