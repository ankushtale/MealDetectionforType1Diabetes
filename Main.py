from Features import Features
from Classifiers import Classifiers

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
np.set_printoptions(precision=4, suppress=True)


# To get all the files from local directory
numFiles = 5
meal = [pd.read_csv('MealNoMealData/mealData{}.csv'.format(i), header=None) for i in range(1, numFiles + 1)]
noMeal = [pd.read_csv('MealNoMealData/Nomeal{}.csv'.format(i), header=None) for i in range(1, numFiles + 1)]


# Transpose
def transpose(df):
    return df.T


for i in range(numFiles):
    meal[i] = transpose(meal[i])
    noMeal[i] = transpose(noMeal[i])


def crosstab_stats(test, pred):
    #pred = list(map(lambda x: 0 if x == 'No Meal' else 1, pred))
    #crosstab = pd.crosstab(test['class'], pred, rownames=['Actual'], colnames=['Predicted'])
    crosstab = pd.crosstab(pred, pred, rownames=['Actual'], colnames=['Predicted'])
    os.linesep
    print("************ Test results ************")
    print("Crosstab: ")
    print(crosstab)
    try:
        tp = crosstab.iloc[0][0]
    except:
        tp = 0
    try:
        fn = crosstab.iloc[0][1]
    except:
        fn = 0
    try:
        fp = crosstab.iloc[1][0]
    except:
        fp = 0
    try:
        tn = crosstab.iloc[1][1]
    except:
        tn = 0
    #tp, fn, fp, tn = crosstab.iloc[0][0], crosstab.iloc[0][1], crosstab.iloc[1][0], crosstab.iloc[1][1]
    accuracy = (tp+tn)/(tp+fn+fp+tn)
    precision = (tp)/(tp+fn)
    recall = (tp)/(tp+fp)
    f1score = 2*precision*recall/(precision + recall)
    os.linesep
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1score: ", f1score)

# Drop column if count NaN is more than Values
def DropIfMaxNaN(df):
    df.dropna(thresh=len(meal[i]) // 2, axis=1, inplace=True)
    df.dropna(thresh=len(noMeal[i]) // 2, axis=1, inplace=True)


# Handling NAN and Reverse
def handlingNaN(df):
    for i in list(df.columns):
        A = np.array(df[i])
        ok = np.logical_not(np.isnan(A))
        xp = ok.ravel().nonzero()[0]
        fp = A[np.logical_not(np.isnan(A))]
        x = np.isnan(A).ravel().nonzero()[0]
        A[np.isnan(A)] = np.interp(x, xp, fp)
        df[i] = pd.DataFrame({'a': list(map(lambda x: round(x, 2), A))})['a']


for i in range(numFiles):
    DropIfMaxNaN(meal[i])
    handlingNaN(meal[i])
    for j in meal[i].columns:
        meal[i][j] = list(meal[i][j])[::-1]
    meal[i].columns = [i for i in range(len(meal[i].columns))]

    DropIfMaxNaN(noMeal[i])
    handlingNaN(noMeal[i])
    for j in noMeal[i].columns:
        noMeal[i][j] = list(noMeal[i][j])[::-1]
    noMeal[i].columns = [i for i in range(len(noMeal[i].columns))]


DeviationFeature = pd.DataFrame(columns=['inRangeCount', 'LowCount', 'HighCount', 'LowMean', 'HighMean', 'Class'])
for i in range(numFiles):
    DeviationFeature = DeviationFeature.append(Features.Deviation(meal[i], 'Meal'), ignore_index=True)
for i in range(numFiles):
    DeviationFeature = DeviationFeature.append(Features.Deviation(noMeal[i], 'No Meal'), ignore_index=True)

mean_range_feature = pd.DataFrame(columns=['MeanRange', 'Class'])
for i in range(numFiles):
    mean_range_feature = mean_range_feature.append(Features.meanRange(meal[i], 'Meal'), ignore_index=True)
for i in range(numFiles):
    mean_range_feature = mean_range_feature.append(Features.meanRange(noMeal[i], 'No Meal'), ignore_index=True)

range_feature = pd.DataFrame(columns=['HighRange', 'LowRange', 'Class'])
for i in range(numFiles):
    range_feature = range_feature.append(Features.Range(meal[i], 'Meal'), ignore_index=True)
for i in range(numFiles):
    range_feature = range_feature.append(Features.Range(noMeal[i], 'No Meal'), ignore_index=True)

fftFeature = pd.DataFrame(columns=['varFFT', 'sdFFT', 'meanFFT', 'Class'])
for i in range(numFiles):
    fftFeature = fftFeature.append(Features.FFT(meal[i], 'Meal'), ignore_index=True)
for i in range(numFiles):
    fftFeature = fftFeature.append(Features.FFT(noMeal[i], 'No Meal'), ignore_index=True)

QuantileFeature = pd.DataFrame(columns=['Quantile', 'Class'])
for i in range(numFiles):
    QuantileFeature = QuantileFeature.append(Features.Quantile(meal[i], 'Meal'), ignore_index=True)
for i in range(numFiles):
    QuantileFeature = QuantileFeature.append(Features.Quantile(noMeal[i], 'No Meal'), ignore_index=True)


# Feature Join
FeatureMatrix = pd.concat(
    [
        DeviationFeature,
        mean_range_feature[['MeanRange']],
        range_feature[['HighRange', 'LowRange']],
        fftFeature[['varFFT', 'sdFFT', 'meanFFT']],
        QuantileFeature['Quantile'],
    ],
    axis=1
)


# TOP FEATURES (EIGEN VECTORS)
def TopFeatures(DF, components):
    x = StandardScaler().fit_transform(FeatureMatrix.drop(['Class'], axis=1))
    pca = decomposition.PCA(n_components=components)
    pca2 = pca.fit(x)

    return list(
        map(
            lambda x: x[1],
            sorted(
                zip(map(lambda x: max(x), pca2.components_), FeatureMatrix.columns),
                key=lambda x: x[0],
                reverse=True
            )[:5]
        )
    )


pca = 0
if pca == 1:
    columns = TopFeatures(FeatureMatrix, len(FeatureMatrix.columns) - 1)
else:
    columns = list(FeatureMatrix.columns)
    columns.remove('Class')


# TRAINING SET
Input = np.array(FeatureMatrix[columns])
Output = np.array(FeatureMatrix['Class'])
inputTrain, inputTest, outputTrain, outputTest = train_test_split(Input, Output, test_size=0.3)

# cv = KFold(n_splits=10, random_state=42, shuffle=False)

# SVM
svc = Classifiers.SVC(inputTrain, outputTrain)
svc_scores = cross_val_score(svc, inputTrain, outputTrain, cv=10)

# linear_model
log = Classifiers.LOG(inputTrain, outputTrain)
log_scores = cross_val_score(log, inputTrain, outputTrain, cv=10)

# neighbors
knn = Classifiers.KNN(inputTrain, outputTrain)
knn_scores = cross_val_score(knn, inputTrain, outputTrain, cv=10)

# RandomForestClassifier
rfc = Classifiers.RFC(inputTrain, outputTrain)
rfc_scores = cross_val_score(rfc, inputTrain, outputTrain, cv=10)

# GaussianNB
gnb = Classifiers.Gaussian(inputTrain, outputTrain)
gnb_scores = cross_val_score(gnb, inputTrain, outputTrain, cv=10)

print("************ Model accuracies ************")
print("Support Vector Machine:\n\t Accuracy: {}\n\t 10-fold CV:{}".format(round(max(svc_scores), 2), svc_scores))
print("K Nearest Neighbor:\n\t Accuracy: {}\n\t 10-fold CV:{}".format(round(max(knn_scores), 2), knn_scores))
print("Logistic Regression:\n\t Accuracy: {}\n\t 10-fold CV:{}".format(round(max(log_scores), 2), log_scores))
print("Random Forest:\n\t Accuracy: {}\n\t 10-fold CV:{}".format(round(max(rfc_scores), 2), rfc_scores))
print("GaussianNB:\n\t Accuracy: {}\n\t 10-fold CV:{}".format(round(max(gnb_scores), 2), gnb_scores))
print()

# predVal = gnb.predict(inputTest)
# trueVal = outputTest
# confusion_matrix(trueVal, predVal)
# print(classification_report(trueVal, predVal))


if __name__ == '__main__':
    fileName = str(input('Enter Test File name:\t'))
    newData = pd.read_csv('{}.csv'.format(fileName), header=None)
    transpose(newData)
    DropIfMaxNaN(newData)
    handlingNaN(newData)

    Test_FeatureMatrix = pd.concat(
        [
            Features.Deviation(newData, 'N/A'),
            Features.meanRange(newData, 'N/A')[['MeanRange']],
            Features.Range(newData, 'N/A')[['HighRange', 'LowRange']],
            Features.FFT(newData, 'N/A')[['varFFT', 'sdFFT', 'meanFFT']],
            Features.Quantile(newData, 'N/A')['Quantile'],
        ],
        axis=1
    )

    if int(input('Pass From PCA 1: YES OR 0: NO:\t')) == 1:
        columns = TopFeatures(Test_FeatureMatrix, len(Test_FeatureMatrix.columns) - 1)
    else:
        columns = list(Test_FeatureMatrix.columns)
        columns.remove('Class')

    Test_DF = Test_FeatureMatrix[columns]

    while True:
        name = str(input('Enter Model Name: SVC, KNN, LOG, RFC, GNB:\t')).upper()
        if name not in ['SVC', 'KNN', 'LOG', 'RFC', 'GNB']:
            break

        model = Classifiers.load(name)
        pred = model.predict(np.array(Test_DF))
        crosstab_stats(pred, pred)
