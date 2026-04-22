# Exploring and Visualizing a Simple Dataset

Exploratory data analysis (EDA) and visualization of the Iris dataset using Python.

## Overview

This project demonstrates basic EDA techniques on the classic Iris flower dataset:
- Loading and inspecting data with pandas
- Generating descriptive statistics
- Creating scatter plots, histograms, and box plots with seaborn/matplotlib

## Requirements

- Python 3.8+
- pandas
- matplotlib
- seaborn

Install dependencies:
```bash
pip install pandas matplotlib seaborn
```

## Usage

Run the script:
```bash
python iris.py
```

The script will:
1. Load the Iris dataset from `iris.csv`
2. Print data shape, columns, and preview rows
3. Display data types and descriptive statistics
4. Generate and display various visualizations

## Dataset

The Iris dataset contains 150 samples from 3 species of iris flowers:
- **Features**: Sepal length, sepal width, petal length, petal width (in cm)
- **Species**: Setosa, Versicolor, Virginica

## Visualizations

| Plot Type | Description |
|-----------|-------------|
| Scatter Plot | Shows relationships between feature pairs, colored by species |
| Histogram | Shows distribution of features with KDE overlay |
| Box Plot | Shows quartiles and potential outliers by species |

## Project Structure

- `iris.py` - Main analysis and visualization script
- `iris.csv` - Iris dataset
- `task.md` - Original task description