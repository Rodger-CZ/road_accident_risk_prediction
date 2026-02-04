# 🚦 Road Accident Risk Prediction

*Kaggle Playground Series – Season 5, Episode 10*
https://www.kaggle.com/competitions/playground-series-s5e10

## 📌 Project Overview

This project presents an end-to-end machine learning pipeline for predicting road accident risk using structured tabular data. The dataset originates from the Kaggle Playground Series (S5E10) and simulates a real-world risk prediction problem.

The goal is to demonstrate practical skills in data preprocessing, model building, evaluation, and deployment in a competition-style workflow.

---

## 🎯 Objectives

* Inspect and clean real-world tabular data
* Build a robust preprocessing pipeline for mixed data types
* Establish a baseline regression model
* Improve performance using gradient boosting
* Evaluate models using cross-validation
* Generate and submit predictions to Kaggle

---

## 🛠️ Tools & Technologies

* **Python**
* **Jupyter Notebook**
* pandas, numpy
* matplotlib, seaborn
* scikit-learn

---

## 📂 Project Structure

```
road-accident-risk-prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── road_accident_risk_prediction.ipynb
│
├── submission.csv
├── README.md
└── requirements.txt
```

---

## 🔍 Methodology

### 1. Data Preparation

* Identified target variable automatically from train/test schema differences
* Handled missing values using appropriate imputation strategies
* Processed numeric and categorical features using `ColumnTransformer`
* Applied one-hot encoding for categorical variables

### 2. Modeling Approach

| Model                         | Description                                           |
| ----------------------------- | ----------------------------------------------------- |
| Ridge Regression              | Baseline linear model with cross-validation           |
| HistGradientBoostingRegressor | Final model capturing non-linear feature interactions |

Cross-validation was used throughout to ensure reliable performance estimation.

---

## 📊 Results

| Model                         | CV RMSE |
| ----------------------------- | ------- |
| Ridge Regression (Baseline)   | ~0.0735 |
| HistGradientBoostingRegressor | ~0.0563 |

### Kaggle Performance

* **Public RMSE:** 0.05591
* **Private RMSE:** 0.05615

The close alignment between public and private scores indicates strong generalization and a robust validation strategy.

---

## 💡 Key Takeaways

* Gradient boosting models significantly outperform linear baselines on tabular risk prediction tasks
* Proper preprocessing and cross-validation are critical for stable model performance
* Small public–private score gaps signal reliable evaluation and low overfitting

---

## 🚀 Future Improvements

* Feature importance analysis and model interpretability
* Hyperparameter tuning (Bayesian / Grid search)
* Experimentation with LightGBM or CatBoost
* Deployment as a lightweight prediction API

---

## 👤 Author

**Faustine Rodgers**
Data Scientist | Machine Learning | Python
