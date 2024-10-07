# House Price Prediction Script Documentation

## Overview
This script implements a machine learning pipeline for predicting house prices using the California Housing dataset. It includes data loading, preprocessing, exploratory data analysis (EDA), feature engineering, and model training.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Data Loading and Initial Exploration
1. Load the dataset from 'housing.csv'.
2. Display basic information about the dataset using `data.info()`.
3. Show summary statistics with `data.describe()`.

## Data Preprocessing
1. Drop missing values using `data.dropna(inplace=True)`.
2. Split the data into features (X) and target variable (y).
3. Further split into training and testing sets using `train_test_split`.

## Exploratory Data Analysis (EDA)
1. Create histograms for all numerical features.
2. Generate a correlation matrix and visualize it using a heatmap.
3. Create a scatter plot of latitude vs longitude, colored by median house value.

## Feature Engineering
1. Create dummy variables for the 'ocean_proximity' categorical feature.
2. Apply log transformation to 'total_bedrooms', 'total_rooms', 'population', and 'households'.
3. Create new features:
   - 'household_ratio': ratio of total rooms to households
   - 'bedroom_ratio': ratio of total bedrooms to total rooms

## Model Training and Evaluation
1. Train a Linear Regression model:
   - Standardize the features using `StandardScaler`.
   - Fit the model on the training data.
   - Evaluate the model on the test set.

2. Train a Random Forest Regressor:
   - Fit the model on the training data.
   - Evaluate the model on the test set.

## Key Functions and Classes Used
- `pd.read_csv()`: Load the dataset
- `train_test_split()`: Split the data into training and testing sets
- `StandardScaler()`: Standardize the features
- `LinearRegression()`: Linear Regression model
- `RandomForestRegressor()`: Random Forest model

## Visualization
The script uses various visualization techniques:
- Histogram plots for feature distribution
- Heatmap for correlation matrix
- Scatter plots for geographical data

## Model Performance
The script evaluates model performance using the R-squared score:
- Linear Regression: Use `lr.score(X_test, y_test)` to get the R-squared value
- Random Forest: Use `forest.score(X_test, y_test)` to get the R-squared value

## Future Improvements
1. Handle missing values more sophisticatedly (e.g., imputation).
2. Experiment with more feature engineering techniques.
3. Try other machine learning algorithms (e.g., Gradient Boosting, Neural Networks).
4. Implement cross-validation for more robust model evaluation.
5. Optimize hyperparameters using techniques like Grid Search or Random Search.
6. Analyze feature importance, especially for the Random Forest model.

## Usage
To run the script:
1. Ensure all dependencies are installed.
2. Place the 'housing.csv' file in the same directory as the script.
3. Run the script using a Python interpreter.

Note: The script currently doesn't save the trained models or predictions. Consider adding functionality to save models and generate predictions for new data.
