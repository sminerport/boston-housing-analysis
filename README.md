# Boston Housing Analysis

This repository contains a comprehensive analysis of the Boston Housing dataset using various regression models, including Linear Regression, Lasso Regression, and Ridge Regression. The project explores the dataset, visualizes the relationships between features and target variables, and evaluates the performance of the different regression models.

## Dataset

The dataset used in this project is the Boston Housing Dataset, which contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. The dataset has 506 samples, with 13 input features and a target variable (MEDV), which represents the median value of owner-occupied homes in $1000's.

**Important Note:** The load_boston function is deprecated in scikit-learn 1.0 and will be removed in 1.2 due to ethical issues related to the dataset. It is strongly recommended to use alternative datasets like the California housing dataset or the Ames housing dataset. To run this Python program without issues, please use a scikit-learn version lower than 1.0.

## Contents

The repository contains a single Python script, `boston_housing_analysis.py`, which includes the following:

1. Importing necessary libraries
2. Loading and exploring the dataset
3. Preprocessing and feature engineering
4. Visualizing the dataset, including histograms, scatterplots, and heatmaps
5. Implementing and evaluating various regression models
6. Calculating and visualizing regression errors and residuals

## Requirements

To run the code, you will need the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Usage

To run the analysis, simply execute the `boston_housing_analysis.py` script in your Python environment:

```bash
python boston_housing_analysis.py
```

The script will output various visualizations, performance metrics, and model coefficients for the different regression models.
