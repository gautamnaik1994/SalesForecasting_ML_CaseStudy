# Sales Forecasting Case Study

- [Sales Forecasting Case Study](#sales-forecasting-case-study)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Tableau Dashboard](#tableau-dashboard)
  - [EDA](#eda)
    - [Uni-variate Analysis](#uni-variate-analysis)
    - [Bi-variate Analysis](#bi-variate-analysis)
      - [Regions by Sales and Orders](#regions-by-sales-and-orders)
      - [Store by Sales and Orders](#store-by-sales-and-orders)
      - [Location by Sales and Orders](#location-by-sales-and-orders)
    - [Multi-variate Analysis](#multi-variate-analysis)
  - [Hypothesis Testing](#hypothesis-testing)
    - [Impact of Discounts on Sales](#impact-of-discounts-on-sales)
    - [Effect of Holidays on Sales](#effect-of-holidays-on-sales)
    - [Sales Differences Across Store](#sales-differences-across-store)
    - [Regional Sales Variability](#regional-sales-variability)
    - [Correlation between Number of Orders and Sales](#correlation-between-number-of-orders-and-sales)
  - [Time Series Analysis](#time-series-analysis)
    - [Trend](#trend)
    - [Seasonality](#seasonality)
      - [Day of Week Seasonality](#day-of-week-seasonality)
      - [Day of the Month Seasonality](#day-of-the-month-seasonality)
      - [Week of the Month Seasonality](#week-of-the-month-seasonality)
      - [Monthly Seasonality](#monthly-seasonality)
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

## Introduction

## Data

## Tableau Dashboard

## EDA

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/gautamnaik1994/SalesForecasting_ML_CaseStudy/blob/main/notebooks/eda/04.TimeSeriesAnalysis.ipynb?flush_cache=true)

### Uni-variate Analysis

### Bi-variate Analysis

#### Regions by Sales and Orders

![Regions](image-5.png)

#### Store by Sales and Orders

![Store by sales and order](image-6.png)

#### Location by Sales and Orders

![LOcation by sales and order](image-7.png)

### Multi-variate Analysis

## Hypothesis Testing

### Impact of Discounts on Sales

Hypothesis: Stores offering discounts will have significantly higher sales than stores not offering discounts

- **Null Hypothesis:**  Stores offering discounts will have the same sales as stores not offering discounts
- **Alternative Hypothesis:** Stores offering discounts will have significantly higher sales than stores not offering discounts

Since p value is 0, it means we can reject the null hypothesis and accept the alternative hypothesis. This means that stores offering discounts will have significantly higher sales than stores not offering discounts.

### Effect of Holidays on Sales

Hypothesis: Sales on holidays are higher compared to non-holidays

- **Null Hypothesis:**  Sales on holidays are the same as sales on non-holidays
- **Alternative Hypothesis:** Sales on holidays are higher compared to non-holidays

Since p value is 1, we fail to reject the null hypothesis. This means that sales on holidays are the same as sales on non-holidays.

### Sales Differences Across Store

Hypothesis: Different store types experience different sales volumes

- **Null Hypothesis:**  Different store types experience the same sales volumes
- **Alternative Hypothesis:** Different store types experience different sales volumes

Since p value is 0, we can reject the null hypothesis and accept the alternative hypothesis. This means that different store types experience different sales volumes.

### Regional Sales Variability

Todo: Add hypothesis testing for regional sales variability

### Correlation between Number of Orders and Sales

![Orders vs Sales](image.png)

- From above plot we can see that there is a positive correlation between number of orders and sales. This means that as the number of orders increase, sales also increase.

## Time Series Analysis

### Trend

### Seasonality

#### Day of Week Seasonality

![Day of Week Seasonality](image-1.png)

- Sales are highest during Saturday and Sunday

#### Day of the Month Seasonality

![Day of the month](image-4.png)

- There is high number of sales during the 5 days of the month

#### Week of the Month Seasonality

![Week of the month](image-3.png)

- There is higher number of sales during the first week of the month
- There is a slight increase in sales during the last week of the month
- The sales is lowest during the 4th week of the month

#### Monthly Seasonality

![Monthly Seasonality](image-2.png)

- The sales is highest during the month of May July, December and January

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
   ![Optuna Dashboard](./timeline.png)
1. Hyperparameter Importance
   ![Optuna Hyperparameter Importance](./parameter_importance.png)

### MLflow

Whenever I work on ML projects, It quickly gets messy with multiple experiments, models,
and hyperparameters. Even the Jupyter notebook gets cluttered with multiple
cells. This is where MLflow comes in. When setup correctly, MLflow can help you
track your experiments, models, and hyperparameters. It also helps you to log
the metrics, parameters, and artifacts. It also helps you to compare the
experiments and models. It also helps you to reproduce the results.

Following are some screenshots of the MLflow UI:

1. Dashboard
   ![MLflow Dashboard](./mlflow.png)

## Deployment

### FastAPI

### Streamlit

## Conclusion
