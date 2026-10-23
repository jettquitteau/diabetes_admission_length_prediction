# Diabetes 130-US Hospitals: Length of Stay Prediction

An end-to-end machine learning pipeline built with scikit-learn to predict
hospital length of stay (`time_in_hospital`, in days) from the UCI Diabetes
130-US Hospitals dataset, which contains about 100,000 hospital encounters
for patients with diabetes.

The focus is on the process of building the pipeline, not on comparing
models, so a single random forest regressor is used.

## Goals

- Clean and preprocess the raw data myself (missing values, categorical
  encoding, grouping of diagnosis codes)
- Build a reproducible scikit-learn pipeline
- Train a random forest regressor
- Evaluate with MAE, RMSE and R²
- Document the decisions made along the way

## Dataset

[UCI Machine Learning Repository: Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Status

Work in progress.
