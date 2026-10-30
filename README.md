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

For this first version, the analysis keeps one row per patient by retaining the
first occurrence in dataset order. This prevents repeat patients from
appearing multiple times, but means the modeling sample represents one selected
encounter per patient rather than all hospital stays.

## Prediction Setup

The intended prediction point is admission. To reduce data leakage, features
that may only be available during or after a stay are excluded, including
procedure and medication counts, secondary diagnoses, discharge disposition,
and readmission status. The primary diagnosis (`diag_1`) is retained as a proxy
for the admitting diagnosis; its availability at the prediction point is an
assumption. The same assumption is true of `medical_specialty`. 

## Prediction Range

In this dataset, `time_in_hospital` ranges from 1 to 14 days. The model is
trained only on this observed target range, so predictions for stays longer
than 14 days are outside its supported range and should not be relied on.

## Status

Work in progress.
