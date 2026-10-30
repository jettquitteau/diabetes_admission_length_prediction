# Diabetes 130-US Hospitals: Length of Stay Prediction

An end-to-end machine learning pipeline built with scikit-learn to predict
hospital length of stay (`time_in_hospital`, in days) from the UCI Diabetes
130-US Hospitals dataset, which contains about 100,000 hospital encounters
for patients with diabetes.

This is a learning project focused on building and evaluating a complete
pipeline, rather than producing a clinically deployable tool. A random forest
regressor is tuned and compared with a simple median-prediction baseline.

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

After cleaning, the modeling dataset contains 71,507 encounters and 11 input
features. Missing categorical values are represented as `Missing`, while the
11 rows with missing primary diagnosis are removed. The weight column is
excluded because of its high missingness.

## Prediction Setup

The intended prediction point is admission. To reduce data leakage, features
that may only be available during or after a stay are excluded, including
procedure and medication counts, secondary diagnoses, discharge disposition,
and readmission status. The primary diagnosis (`diag_1`) is retained as a proxy
for the admitting diagnosis; its availability at the prediction point is an
assumption. The same assumption is true of `medical_specialty`. 

## Prediction Range

In this dataset, `time_in_hospital` ranges from 1 to 14 days. The random forest
averages training targets and cannot extrapolate beyond their range. It cannot
predict stays longer than 14 days, and this evaluation provides no evidence
about performance on those encounters.

## Method

- Split the cleaned data into 80% training and 20% test data using
  `random_state=42`.
- Fit categorical preprocessing within a scikit-learn pipeline, using
  one-hot encoding and grouping categories below 1% frequency. These
  frequencies are learned from the training portion of each fit, not the test set.
- Compare against a `DummyRegressor` that always predicts the training median.
- Tune a 100-tree random forest using five-fold cross-validation on the training
  set, optimizing MAE over `min_samples_leaf` and `max_features`.
- Evaluate the selected model on the held-out test set using MAE, RMSE, and R2,
  then inspect actual-versus-predicted values and permutation importance.

The selected settings were `min_samples_leaf=20` and `max_features=0.6`, with
a cross-validation MAE of approximately 2.17 days.

## Results

| Model / Split | MAE (days) | RMSE (days) | R2 |
| --- | ---: | ---: | ---: |
| Median baseline / test | 2.207 | 3.189 | -0.197 |
| Tuned random forest / train | 2.09 | 2.72 | 0.155 |
| Tuned random forest / test | 2.13 | 2.78 | 0.093 |

The random forest improves on the baseline, but the improvement in MAE is
modest: approximately 0.08 days. Its test R2 of 0.093 indicates that it explains
only about 9.3% of the variation in length of stay on the test set. MAE describes
an average error, not a guarantee of accuracy for an individual encounter.

### Longer Stays

The actual-versus-predicted plot in the
[notebook](diabetes_length_of_admission.ipynb) shows predictions concentrated
around the middle of the observed range. Longer actual stays, particularly
those approaching 10-14 days, are substantially underpredicted. This failure
occurs even within the dataset's supported range, separately from the inability
to predict beyond 14 days.

This limits the model's practical usefulness. Identifying patients likely to
have prolonged stays could be especially valuable for bed management and
discharge planning, yet these are precisely the encounters for which the plot
shows poor predictions. Aggregate metrics alone do not capture this weakness,
it is only suggested by the graph.

## Interpretation and Limitations

- The model is not suitable for clinical or operational decision-making in its
  current form.
- Restricting inputs to information assumed available at admission reduces
  leakage risk but leaves out much of the information describing the hospital
  course. The availability of `diag_1` and `medical_specialty` remains an assumption.
- The feature importance shows that `diag_1` is by far the most important feature,
  which is not surprising given the limited infomation available. As its availability
  is an assumption, it remains another large limitation to this model as that
  information is not necessarily something that would be available at admission.
  `age` and `admission_source_id` are the next two, which is unsurprising as well.
- The right-skewed target and predictions concentrated toward typical stays are
  consistent with difficulty identifying longer stays.
- The data used was collected two decades ago, increasing the difficulty in
  generalizing for today were there to be an attempt to actually deploy the model.
- Missing categories may capture documentation practices rather than patient
  characteristics.

## Learning Outcomes

The main outcome is a documented workflow covering data cleaning, prediction-time
assumptions, leakage prevention, pipeline-based preprocessing, baseline comparison,
cross-validated tuning, and held-out evaluation. A key takeaway is that beating
a baseline does not necessarily make a model useful.

Possible next steps include quantifying errors by stay length, comparing other
models, and assessing additional predictors whose availability at admission can
be verified.

## Status

Completed as an initial learning project; not meant for deployment.

Jett Guitteau
October 2026