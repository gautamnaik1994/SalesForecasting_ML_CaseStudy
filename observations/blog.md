# Sales Forecasting

## Introduction

## Data

## Tableau Dashboard

## EDA

## Hypothesis Testing

## Time Series Analysis

## Prediction and Forecasting

### Machine Learning Models

1. Linear Regression
1. Random Forest
1. XGBoost
1. LightGBM

#### Feature Engineering

#### Metric

I used Mean Absolute Error (MAE) as the evaluation metric for the models.

### Plots

1. Feature Importance
1. Residuals
1. Actual vs. Predicted

### Time Series Models

1. Triple Exponential Smoothing (Holt-Winters)
1. ARIMA
1. SARIMA
1. SARIMAX
1. MSTLES

#### Feature Engineering

#### Metric

I used Mean Absolute Percentage Error (MAPE) as the evaluation metric for the models.

### Plots

1. Feature Importance
1. Residuals
1. Actual vs. Predicted

## Hyperparameter Tuning

Hyperparameter tuning is the process of finding the best hyperparameters for a
model. Grid Search and Random Search are the most common methods for
hyperparameter tuning. However, these methods are computationally expensive and
time-consuming.This is the reason, I chose [Optuna](https://optuna.org/) for hyperparameter tuning.
Along with Optuna, I used [MLflow](https://mlflow.org/) for tracking the experiments.
I have explained in detail about MLflow and Optuna in the next section.

### Optuna

Optuna is a hyperparameter optimization framework that uses
[Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) to
find the best hyperparameters for a model. Optuna optially saves information
about all the trails in a local SQLite database. It also provides a web UI to
visualize the trails. Using this web UI, we can compare diffrent trails and also
shows which hyperparameters are important.

Following are some screenshots of the Optuna UI:

1. Dashboard
1. Parallel Coordinate Plot
1. Hyperparameter Importance
1. Hyperparameter Distribution
1. Hyperparameter Correlation

### MLflow

Whenever I work on ML projects, It quickly gets messy with multiple experiments, models,
and hyperparameters. Even the Jupyter notebook gets cluttered with multiple
cells. This is where MLflow comes in. When setup correctly, MLflow can help you
track your experiments, models, and hyperparameters. It also helps you to log
the metrics, parameters, and artifacts. It also helps you to compare the
experiments and models. It also helps you to reproduce the results.

Following are some screenshots of the MLflow UI:

1. Experiments
1. Runs
1. Parameters
1. Metrics
1. Artifacts

## Deployment

### FastAPI

### Streamlit

## Conclusion
