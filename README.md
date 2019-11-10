# Meal Detection from Continous Glucose Monitor Data

One of the problem statements in automated insulin injetion for Type 1 Diabeties treament is determining the quantity and timing for ingestion of the insulin. In this exercise, we contribute towards prediction of timing of ingestion by guessing meal intake from Continous Glucose Monitor's data.


### Objective
Given multiple timeseries from CGM data for a 2.5 hour window, our objective is to predict wheather the timeseries belongs to a meal intake or not.


### Implementation Details

**Models used**
1. Support Vector Machines
2. K-Nearest Neighbor
3. Logistic Regression
4. Random Forest
5. GaussianNB

**Salient features**:
- Data clean-up (Handle columns with >50% NaN values)
- Pre-processing (Combination of Forward fill, Backward fill and Range Fill)
- Grid Search for Hyperparameter Tuning
- Principal Component Analysis for Dimensionality Reduction; suggest top features
- Save and load model weights
- Modular application - seperation of concern


### Pre-requisite
1. Make sure _scikit-learn, pandas, numpy and pickle_ are installed in your environment.
2. Change to repo directory
3. Make sure the test data doesn't have Empty rows or unequal lengths of rows. If there are unequal rows, pad them with _NaN_
4. Run main.py
```python
python main.py
```

**Test Data format**:
- 1 csv file with similar structure as that of train data. Refer to Train Data.
- 1 ground truth file in csv. Use 1 for Meal and 0 for Nomeal in one-column format.

Note: While rows with >50% NaNs have been excluded from training, avoid passing such inputs


### Expected Output
- Accuracies of models on training data
- Scores from 10-fold Cross Validation for each model
- <_input_> Test File Name (present in same working directory)
- <_input_> Choose to reduce dimensions of test data (PCA: Yes or No)
- <_input_> Choose Model


### Contributors

|ASU id | Names        | Features           | Model  | Work |
| -----| :-------------: |:-------------:| -----:| -----:|
|1215185400| Vaibhav Singhal      | Deviation, Range | Random Forest | Code, modularity of code, testing script |
|1217124012| Ankush Tale      | Quantiles, Mann Kendall Trend Test      |   Logistic Regression | Accuracy measures, refactoring code, K-fold Cross Validation |
|1217202363| Keshin Jani | MeanRange |    GaussianNB, Support Vector | feature function, feature exploration |
|1215146842| Manikanta Chinkunta | MeanRange |   Variance FFT, Mean FFT, Standard Deviation FFT|  KNearestNeighbor | Feature function |

1. Vaibhav Singhal
* Features: Deviation, Range
* Work: Code, modularity of code, testing script
* Model: Random Forest 
2. Ankush Tale
* Features: Quantile, Mann Kendall Trend Test
* Work: Accuracy measures, refactoring code, K-fold Cross Validation
* Model: Logistic Regression, Support Vector
3. Keshin Jani
* Features: MeanRange
* Work: feature function
* Model: GaussianNB, Support Vector
4. Manikanta Chintakunta
* Features: Variance FFT, Mean FFT, Standard Deviation FFT
* Work: Feature function
* Model: KNearestNeighbor

