- [Sales Forecasting](#sales-forecasting)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Tableau Dashboard](#tableau-dashboard)
  - [EDA](#eda)
  - [Hypothesis Testing](#hypothesis-testing)
  - [Time Series Analysis](#time-series-analysis)
  - [Prediction and Forecasting](#prediction-and-forecasting)
    - [Machine Learning Models](#machine-learning-models)
      - [Feature Engineering](#feature-engineering)
      - [Metric](#metric)
    - [Plots](#plots)
    - [Time Series Models](#time-series-models)
      - [Feature Engineering](#feature-engineering-1)
      - [Metric](#metric-1)
    - [Plots](#plots-1)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Optuna](#optuna)
    - [MLflow](#mlflow)
  - [Deployment](#deployment)
    - [FastAPI](#fastapi)
    - [Streamlit](#streamlit)
  - [Conclusion](#conclusion)

# Sales Forecasting

## Introduction

## Data

## Tableau Dashboard

## EDA

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/gautamnaik1994/SalesForecasting_ML_CaseStudy/blob/main/notebooks/eda/04.TimeSeriesAnalysis.ipynb?flush_cache=true)

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
   ![Optuna Dashboard]("./timeline.png")
1. Hyperparameter Importance
   ![Optuna Hyperparameter Importance]("./parameter_importance.png")

### MLflow

Whenever I work on ML projects, It quickly gets messy with multiple experiments, models,
and hyperparameters. Even the Jupyter notebook gets cluttered with multiple
cells. This is where MLflow comes in. When setup correctly, MLflow can help you
track your experiments, models, and hyperparameters. It also helps you to log
the metrics, parameters, and artifacts. It also helps you to compare the
experiments and models. It also helps you to reproduce the results.

Following are some screenshots of the MLflow UI:

1. Dashboard
   ![MLflow Dashboard]("/mlflow.png")

## Deployment

### FastAPI

### Streamlit

## Conclusion
