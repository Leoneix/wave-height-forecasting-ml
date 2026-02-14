---
layout: post
title: "Significant Wave Height Forecasting Using Comparative Machine Learning Models"
author: Your Name
date: 2026-02-14
categories: [Machine Learning, Time Series, Ocean Modeling]
---

## Abstract

This study investigates short-term forecasting of Significant Wave Height (WVHT) using multivariate buoy-based meteorological and oceanographic measurements. A persistence baseline, linear regression, ensemble tree-based models (Random Forest, XGBoost), and an LSTM neural network were evaluated under strict chronological time-series validation.

Results indicate that gradient boosting achieved the highest performance (R² ≈ 0.97), significantly outperforming the persistence baseline and LSTM architecture. The findings suggest that structured ensemble methods are highly effective for oceanographic time-series with strong short-term inertia and seasonal structure.

## 1. Introduction

Accurate forecasting of ocean wave height is critical for maritime navigation, offshore engineering, and coastal risk assessment. Traditional numerical models rely on physical equations, but data-driven approaches offer computationally efficient alternatives for short-term forecasting.

This work evaluates whether modern machine learning models, including ensemble methods and deep learning architectures, can outperform classical baselines in predicting Significant Wave Height from buoy-based meteorological observations.

## 2. Dataset Description

The dataset consists of time-series measurements recorded by ocean buoys. Variables include wind speed, wind direction, atmospheric pressure, wave period, air temperature, and water temperature.

The target variable is Significant Wave Height (WVHT), measured in meters.

All timestamp components were combined into a unified datetime index and sorted chronologically to preserve temporal integrity.

| Variable | Description                 |
|----------|-----------------------------|
| WSPD     | Wind Speed (m/s)            |
| PRES     | Atmospheric Pressure (hPa)  |
| DPD      | Dominant Wave Period (sec)  |
| WVHT     | Significant Wave Height (m) |

## 3. Exploratory Analysis

Seasonal analysis revealed consistent calm wave conditions between July and September each year. Autocorrelation analysis showed strong lag-1 dependency, indicating significant short-term inertia in wave height dynamics.

These findings suggest that persistence provides a strong baseline and that model improvements must capture nonlinear meteorological interactions rather than simple memory effects.

### 4.1 Time-Series Validation

A chronological 80-20 train-test split was implemented to prevent data leakage. Random shuffling was strictly avoided.

### 4.2 Persistence Baseline

The persistence model assumes:

$$
WVHT_t = WVHT_{t-1}
$$

This served as the minimum benchmark for all models.

### 4.3 Machine Learning Models

The following models were evaluated:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LSTM (24-hour sequence window)

Tree-based models utilized lag and seasonal encoding features. The LSTM model was trained on scaled multivariate sequences.

## 5. Results

| Model | R² Score |
|--------|----------|
| Linear Regression | 0.9599 |
| Random Forest | 0.9650 |
| XGBoost | **0.9698** |
| LSTM | 0.8875 |


## 6. Discussion

The superior performance of XGBoost indicates that nonlinear feature interactions between wind speed, pressure, and wave period are critical predictors of wave height.

The underperformance of LSTM suggests that long-range sequence modeling provides limited additional benefit in the presence of strong short-term inertia and well-engineered lag features.

These findings emphasize the importance of model selection based on data structure rather than architectural complexity.

## 7. Conclusion

This study demonstrates that ensemble tree-based methods outperform deep sequential models for short-term significant wave height forecasting in structured buoy datasets.

Future work includes multi-step forecasting and extreme event modeling.

---


