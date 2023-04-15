# Boston Housing Analysis

This repository contains a comprehensive analysis of the Boston Housing dataset using various regression models, including Linear Regression, Lasso Regression, and Ridge Regression. The project explores the dataset, visualizes the relationships between features and target variables, and evaluates the performance of the different regression models.

## Dataset

The dataset used in this project is the Boston Housing Dataset, which contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. The dataset has 506 samples, with 13 input features and a target variable (MEDV), which represents the median value of owner-occupied homes in $1000's.

**Important Note:** The `load_boston` function is deprecated in scikit-learn 1.0 and will be removed in 1.2 due to ethical issues related to the dataset. It is strongly recommended to use alternative datasets like the California housing dataset or the Ames housing dataset. To run this Python program without issues, please use a scikit-learn version lower than 1.0.

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
- pandas (version 1.5.3)
- matplotlib
- seaborn
- scikit-learn (version lower than 1.0)

**Note:** This project uses pandas version 1.5.3 instead of the latest version (2.0.0) because the `append()` function is deprecated and replaced by `concat()` in version 2.0.0. For more information, check the [pandas 2.0.0 release notes](https://pandas.pydata.org/docs/whatsnew/v2.0.0.html#deprecations).

The repository includes a `requirements.txt` file that lists the necessary library versions. After cloning the repository and navigating to the project root folder (boston-housing-analysis), run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the analysis, simply execute the `boston_housing_analysis.py` script in your Python environment:

```bash
python boston_housing_analysis.py
```

The script will output various visualizations, performance metrics, and model coefficients for the different regression models.

If you want to use an alternative dataset like the California housing dataset or the Ames housing dataset, please follow the instructions below and modify the code accordingly:

For the California housing dataset:

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
```

For the Ames housing dataset:
```python
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)
```

To use one of these alternative datasets, replace the code that loads the Boston Housing dataset in the boston_housing_analysis.py script with one of the code snippets above. Please note that these alternative datasets have different features, so you may need to update the feature engineering, preprocessing, and visualization sections of the script accordingly.
