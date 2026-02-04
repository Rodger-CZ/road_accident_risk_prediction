# %% [markdown]
# # Road Accident Risk Prediction — End-to-End ML Pipeline
# 
# This notebook presents a complete machine learning workflow for predicting road accident risk using structured tabular data from the Kaggle Playground Series S5E10 competition.
# 
# ## What this notebook covers
# 
# Data loading and quality assessment
# 
# Robust preprocessing for numeric and categorical features
# 
# Baseline modeling using Ridge Regression with cross-validation
# 
# Model improvement using HistGradientBoostingRegressor
# 
# Proper model evaluation using RMSE
# 
# Generation of a valid Kaggle submission
# 
# ## Key highlights
# 
# Cross-validation was used throughout to ensure reliable performance estimates
# 
# Gradient boosting significantly outperformed the linear baseline
# 
# The final model achieved strong and stable results with minimal public–private score gap
# 
# # Results
# 
# Public RMSE: 0.05591
# 
# Private RMSE: 0.05615
# 
# The close alignment between validation, public, and private scores indicates good generalization and a robust modeling approach.
# 
# This notebook is intended to demonstrate practical skills in tabular machine learning, model evaluation, and competition-style workflows.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:32.321270Z","iopub.execute_input":"2026-02-04T07:09:32.321531Z","iopub.status.idle":"2026-02-04T07:09:32.327581Z","shell.execute_reply.started":"2026-02-04T07:09:32.321511Z","shell.execute_reply":"2026-02-04T07:09:32.326554Z"},"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Road Accident Risk Prediction
# *Kaggle Playground Series – Season 5, Episode 10*
# 
# ## Project Overview
# This project builds a machine learning pipeline to predict road accident risk
# using structured tabular data. The workflow follows a real-world ML process:
# data inspection, cleaning, exploratory analysis, feature preprocessing,
# model training with cross-validation, and Kaggle submission generation.
# 
# ## Objectives
# - Understand and clean the dataset
# - Identify important features and patterns
# - Build baseline and improved regression models
# - Evaluate performance using cross-validation
# - Generate a valid Kaggle submission
# 
# ## Tools
# Python, pandas, numpy, matplotlib, seaborn, scikit-learn

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:32.344466Z","iopub.execute_input":"2026-02-04T07:09:32.344846Z","iopub.status.idle":"2026-02-04T07:09:32.352302Z","shell.execute_reply.started":"2026-02-04T07:09:32.344822Z","shell.execute_reply":"2026-02-04T07:09:32.351345Z"},"jupyter":{"outputs_hidden":false}}
# We import the necessary Libraries and Packages
# For Data Handling
import pandas as pd  
import numpy as np

# For EDA Visuals
import matplotlib.pyplot as plt
import seaborn as sns

# For regression, Feature engineering, and cross validation
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge

sns.set_palette("Set2")
plt.style.use("seaborn-v0_8")
RANDOM_STATE = 42

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:32.358360Z","iopub.execute_input":"2026-02-04T07:09:32.358835Z","iopub.status.idle":"2026-02-04T07:09:33.057263Z","shell.execute_reply.started":"2026-02-04T07:09:32.358797Z","shell.execute_reply":"2026-02-04T07:09:33.056432Z"},"jupyter":{"outputs_hidden":false}}
# We load the datasets into our notebook
# We read the Train dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
# Test Dataset
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
# Submission Dataset
sub  = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

# We inspect the Train Datasets - rows and targets
# Test Dataset: same features, no target
# Submission: id + target column
train.shape, test.shape, sub.shape

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.058471Z","iopub.execute_input":"2026-02-04T07:09:33.058674Z","iopub.status.idle":"2026-02-04T07:09:33.075900Z","shell.execute_reply.started":"2026-02-04T07:09:33.058654Z","shell.execute_reply":"2026-02-04T07:09:33.074923Z"},"jupyter":{"outputs_hidden":false}}
# We get an overview of the dataset
# We are going to use the Train Dataset
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.077023Z","iopub.execute_input":"2026-02-04T07:09:33.077322Z","iopub.status.idle":"2026-02-04T07:09:33.222532Z","shell.execute_reply.started":"2026-02-04T07:09:33.077295Z","shell.execute_reply":"2026-02-04T07:09:33.221870Z"},"jupyter":{"outputs_hidden":false}}
train.info()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.223960Z","iopub.execute_input":"2026-02-04T07:09:33.224153Z","iopub.status.idle":"2026-02-04T07:09:33.228303Z","shell.execute_reply.started":"2026-02-04T07:09:33.224134Z","shell.execute_reply":"2026-02-04T07:09:33.227657Z"},"jupyter":{"outputs_hidden":false}}
# We Identify the target column
set(train.columns) - set(test.columns)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# Our target column is 'accident_risk'

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.229008Z","iopub.execute_input":"2026-02-04T07:09:33.229211Z","iopub.status.idle":"2026-02-04T07:09:33.248414Z","shell.execute_reply.started":"2026-02-04T07:09:33.229186Z","shell.execute_reply":"2026-02-04T07:09:33.247615Z"},"jupyter":{"outputs_hidden":false}}
# We assign the target column that we found above
TARGET = list(set(train.columns) - set(test.columns))[0]
TARGET

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.249924Z","iopub.execute_input":"2026-02-04T07:09:33.250175Z","iopub.status.idle":"2026-02-04T07:09:33.276813Z","shell.execute_reply.started":"2026-02-04T07:09:33.250145Z","shell.execute_reply":"2026-02-04T07:09:33.275814Z"},"jupyter":{"outputs_hidden":false}}
# We confirm the distribution for the target column
train["accident_risk"].value_counts(normalize=True)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.278143Z","iopub.execute_input":"2026-02-04T07:09:33.278483Z","iopub.status.idle":"2026-02-04T07:09:33.329186Z","shell.execute_reply.started":"2026-02-04T07:09:33.278454Z","shell.execute_reply":"2026-02-04T07:09:33.328321Z"},"jupyter":{"outputs_hidden":false}}
# We move to separate the features from the Target
X = train.drop(columns=[TARGET])
y = train[TARGET]

X.shape, y.shape

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Data Quality Assessment
# The dataset was inspected for missing values, duplicates, and inconsistent
# data types to ensure reliable modeling.
# 
# Note: The preprocessing pipeline uses scikit-learn's ColumnTransformer
# to handle numeric and categorical features robustly and is compatible
# with recent scikit-learn versions.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.330111Z","iopub.execute_input":"2026-02-04T07:09:33.330333Z","iopub.status.idle":"2026-02-04T07:09:33.435899Z","shell.execute_reply.started":"2026-02-04T07:09:33.330310Z","shell.execute_reply":"2026-02-04T07:09:33.435047Z"},"jupyter":{"outputs_hidden":false}}
# We check for null values 
X.isna().sum().sort_values(ascending=False).head(10)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# There are no missing values in our dataset

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.436744Z","iopub.execute_input":"2026-02-04T07:09:33.436997Z","iopub.status.idle":"2026-02-04T07:09:33.620983Z","shell.execute_reply.started":"2026-02-04T07:09:33.436970Z","shell.execute_reply":"2026-02-04T07:09:33.620354Z"},"jupyter":{"outputs_hidden":false}}
# We check for Duplicates in our dataset
train.duplicated().sum()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# There are also no duplicates in our dataset

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.623337Z","iopub.execute_input":"2026-02-04T07:09:33.623536Z","iopub.status.idle":"2026-02-04T07:09:33.728505Z","shell.execute_reply.started":"2026-02-04T07:09:33.623517Z","shell.execute_reply":"2026-02-04T07:09:33.727858Z"},"jupyter":{"outputs_hidden":false}}
# Numerical Feature overview 
num_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()

train[num_cols].describe().T

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# We are watching out for:
# 
# Unrealistic Values
# 
# Heavy skew
# 
# Outliers

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.729243Z","iopub.execute_input":"2026-02-04T07:09:33.729418Z","iopub.status.idle":"2026-02-04T07:09:33.888047Z","shell.execute_reply.started":"2026-02-04T07:09:33.729400Z","shell.execute_reply":"2026-02-04T07:09:33.886091Z"},"jupyter":{"outputs_hidden":false}}
# Categorical Feature Overview
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    print(f"\n{col}")
    print(train[col].value_counts().head())

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# This is mainly for: 
# 
# Cardinality check
# 
# Rare categories
# 
# Encoding strategy planning

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.889267Z","iopub.execute_input":"2026-02-04T07:09:33.889547Z","iopub.status.idle":"2026-02-04T07:09:33.913041Z","shell.execute_reply.started":"2026-02-04T07:09:33.889523Z","shell.execute_reply":"2026-02-04T07:09:33.912151Z"},"jupyter":{"outputs_hidden":false}}
# We are going to identify numeric and categorical variables
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

len(num_features), len(cat_features)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.914021Z","iopub.execute_input":"2026-02-04T07:09:33.914302Z","iopub.status.idle":"2026-02-04T07:09:33.920414Z","shell.execute_reply.started":"2026-02-04T07:09:33.914277Z","shell.execute_reply":"2026-02-04T07:09:33.919667Z"},"jupyter":{"outputs_hidden":false}}
num_features[:5], cat_features[:5]

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.921173Z","iopub.execute_input":"2026-02-04T07:09:33.921337Z","iopub.status.idle":"2026-02-04T07:09:33.935576Z","shell.execute_reply.started":"2026-02-04T07:09:33.921321Z","shell.execute_reply":"2026-02-04T07:09:33.934734Z"},"jupyter":{"outputs_hidden":false}}
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Baseline Model
# A Ridge Regression model was used as a baseline to establish a reference
# performance using cross-validation.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:33.936558Z","iopub.execute_input":"2026-02-04T07:09:33.936844Z","iopub.status.idle":"2026-02-04T07:09:38.703500Z","shell.execute_reply.started":"2026-02-04T07:09:33.936817Z","shell.execute_reply":"2026-02-04T07:09:38.702829Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import root_mean_squared_error

model = Ridge(random_state=RANDOM_STATE)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rmse_scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    rmse = root_mean_squared_error(y_val, preds)
    rmse_scores.append(rmse)

print("CV RMSE mean:", np.mean(rmse_scores))
print("CV RMSE std :", np.std(rmse_scores))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Target Exploration
# Before improving models, we inspect the target distribution to check for skewness
# and decide whether a transformation might improve performance.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:38.704650Z","iopub.execute_input":"2026-02-04T07:09:38.704920Z","iopub.status.idle":"2026-02-04T07:09:38.965768Z","shell.execute_reply.started":"2026-02-04T07:09:38.704897Z","shell.execute_reply":"2026-02-04T07:09:38.965065Z"},"jupyter":{"outputs_hidden":false}}
# We plot a chart to show the distribution on the target distribution
plt.figure(figsize=(8,4))
plt.hist(y, bins=50)
plt.title("Target Distribution")
plt.xlabel(TARGET)
plt.ylabel("Count")
plt.tight_layout()
plt.show()

y.describe()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:38.966662Z","iopub.execute_input":"2026-02-04T07:09:38.966975Z","iopub.status.idle":"2026-02-04T07:09:38.976010Z","shell.execute_reply.started":"2026-02-04T07:09:38.966937Z","shell.execute_reply":"2026-02-04T07:09:38.975172Z"},"jupyter":{"outputs_hidden":false}}
# We are checking the skew
y.skew()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# STEP 13 — Stronger model (recommended): HistGradientBoostingRegressor
# 
# This is fast, strong, and built into sklearn (no extra installs).
# It also handles non-linear patterns much better than Ridge.
# 
# Important note
# 
# HistGradientBoosting works only with numeric features, so we’ll:
# 
# impute numerics
# 
# one-hot encode categoricals (still numeric)
# 
# and feed into the model

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:38.977494Z","iopub.execute_input":"2026-02-04T07:09:38.977926Z","iopub.status.idle":"2026-02-04T07:09:38.988339Z","shell.execute_reply.started":"2026-02-04T07:09:38.977888Z","shell.execute_reply":"2026-02-04T07:09:38.987627Z"},"jupyter":{"outputs_hidden":false}}
# Defining the model pipeline 
from sklearn.ensemble import HistGradientBoostingRegressor

hgb_model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=500,
    random_state=RANDOM_STATE
)

hgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", hgb_model)
])

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:09:38.989209Z","iopub.execute_input":"2026-02-04T07:09:38.989411Z","iopub.status.idle":"2026-02-04T07:10:29.824506Z","shell.execute_reply.started":"2026-02-04T07:09:38.989390Z","shell.execute_reply":"2026-02-04T07:10:29.823757Z"},"jupyter":{"outputs_hidden":false}}
# Cross validation (Comparing to Ridger Baseline)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rmse_scores_hgb = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    hgb_pipeline.fit(X_train, y_train)
    preds = hgb_pipeline.predict(X_val)

    rmse = root_mean_squared_error(y_val, preds)
    rmse_scores_hgb.append(rmse)

print("HGB CV RMSE mean:", np.mean(rmse_scores_hgb))
print("HGB CV RMSE std :", np.std(rmse_scores_hgb))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Model Selection
# The HistGradientBoostingRegressor significantly outperformed the baseline Ridge
# Regression model, reducing cross-validated RMSE from ~0.073 to ~0.056.
# 
# Given its superior performance and stability, this model was selected for final
# training and submission.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# We Train our final model on full data and generate the submission

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:10:29.825370Z","iopub.execute_input":"2026-02-04T07:10:29.825538Z","iopub.status.idle":"2026-02-04T07:10:43.710383Z","shell.execute_reply.started":"2026-02-04T07:10:29.825521Z","shell.execute_reply":"2026-02-04T07:10:43.709497Z"},"jupyter":{"outputs_hidden":false}}
hgb_pipeline.fit(X, y)
test_preds = hgb_pipeline.predict(test)

submission = sub.copy()
submission[TARGET] = test_preds
submission.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:10:43.711270Z","iopub.execute_input":"2026-02-04T07:10:43.711453Z","iopub.status.idle":"2026-02-04T07:10:43.977667Z","shell.execute_reply.started":"2026-02-04T07:10:43.711435Z","shell.execute_reply":"2026-02-04T07:10:43.976812Z"},"jupyter":{"outputs_hidden":false}}
# We are going to save the submission
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-04T07:10:43.979840Z","iopub.execute_input":"2026-02-04T07:10:43.980109Z","iopub.status.idle":"2026-02-04T07:10:43.987116Z","shell.execute_reply.started":"2026-02-04T07:10:43.980090Z","shell.execute_reply":"2026-02-04T07:10:43.985232Z"},"jupyter":{"outputs_hidden":false}}
submission.head()
submission.shape

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Results Summary
# 
# | Model | CV RMSE |
# |------|--------|
# | Ridge Regression (Baseline) | ~0.073 |
# | HistGradientBoostingRegressor | ~0.056 |
# 
# The final model achieved a substantial improvement over the baseline,
# demonstrating the importance of non-linear models for tabular risk prediction
# tasks.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Kaggle Results
# 
# The final model was submitted to the Kaggle Playground Series S5E10 competition.
# 
# - **Public RMSE:** 0.05591
# - **Private RMSE:** 0.05615
# 
# The close alignment between public and private scores indicates good model
# generalization and a robust validation strategy.
