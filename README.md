# Meal Detection from Continous Glucose Monitor Data

One of the problem statements in automated insulin injetion for Type 1 Diabeties treament is determining the quantity and timing for ingestion of the insulin. In this exercise, we contribute towards prediction of timing of ingestion by guessing meal intake from Continous Glucose Monitor's data.


## Objective
Given multiple timeseries from CGM data for a 2.5 hour window, our objective is to predict wheather the timeseries belongs to a meal intake or not.


## Implementation Details

**Models used with accuracies**
1. Logistic Regression (91%)
2. Random Forest (82%)
3. GaussianNB (67%)
4. Support Vector Machines (54%)
5. K-Nearest Neighbor (61%)
![alt text](https://github.com/ankushtale/MealDetectionforType1Diabetes/docs/model_accuracies.jpg "Model Accuracies")

**Salient features**:
- Data clean-up (Handle columns with >50% NaN values)
- Pre-processing (Combination of Forward fill, Backward fill and Range Fill)
- Grid Search for Hyperparameter Tuning
- Principal Component Analysis for Dimensionality Reduction; suggest top features
- Save and load model pickle file with model and weights
- Modular application - seperation of concern


## Run steps
1. Make sure _scikit-learn, pandas, numpy and pickle_ are installed in your environment.
2. Change to repo directory
3. Make sure the test data doesn't have Empty rows or unequal lengths of rows. If there are unequal rows, pad them with _NaN_
4. Run train.py
```python
python train.py
```
This will save weights in .sav files in working directory
5. Run test.py
```python
python test.py
```
6. If you train data with PCA, enter 'YES'/1 when prompted for PCA in test.py


**Test Data format**:
- 1 csv file with similar structure as that of train data. Refer to Train Data.
- 1 ground truth file in csv. Use 1 for Meal and 0 for Nomeal in one-column format.

Note: While rows with >50% NaNs have been excluded from training, avoid passing such inputs


## Expected Output
- Accuracies of models on training data
- Scores from 10-fold Cross Validation for each model
- <_input_> Test File Name (present in same working directory)
- <_input_> Choose to reduce dimensions of test data (PCA: Yes or No)
- <_input_> Choose Model


## Contributors

All models were randomly picked by team members for implementation

|ASU id | Names        | Features           | Model  | Work |
| -----| :-------------: |:-------------:| -----:| -----:|
|1215185400| Vaibhav Singhal      | Deviation, Range | Random Forest | Code, modularity of code, testing script |
|1217124012| Ankush Tale      | Quantiles, Mann Kendall Trend Test      |   Logistic Regression | Accuracy measures, refactoring code, K-fold Cross Validation |
|1217202363| Keshin Jani | MeanRange |    GaussianNB, Support Vector | feature function, feature exploration |
|1215146842| Manikanta Chinkunta | MeanRange |   Variance FFT, Mean FFT, Standard Deviation FFT|  KNearestNeighbor | Feature function |
