from features import Features
from classifiers import Classifiers

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from warnings import simplefilter

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
np.set_printoptions(precision=4, suppress=True)
simplefilter(action='ignore', category=FutureWarning)


def crosstab_stats(test, pred):
    crosstab = pd.crosstab(test, pred, rownames=['Actual'], colnames=['Predicted'])
    print()
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
    print(tp, fp)
    print(fn, tn)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    precision = (tp) / (tp + fn)
    recall = (tp) / (tp + fp)
    f1score = 2 * precision * recall / (precision + recall)
    print()
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1score: ", f1score)


# Transpose
def transpose(df):
    return df.T


# Drop column if count NaN is more than Values
def DropIfMaxNaN(df):
    #for i in df.coluns:
    df.dropna(thresh=len(df[0]) // 2, axis=1, inplace=True)


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


# TOP FEATURES (EIGEN VECTORS)
def TopFeatures(df, components):
    x = StandardScaler().fit_transform(df.drop(['Class'], axis=1))
    pca = decomposition.PCA(n_components=components)
    pca2 = pca.fit(x)

    return list(
        map(
            lambda x: x[1],
            sorted(
                zip(map(lambda x: max(x), pca2.components_), df.columns),
                key=lambda x: x[0],
                reverse=True
            )[:5]
        )
    )


if __name__ == '__main__':
    # To get all the files from local directory
    numFiles = 5
    meal = [pd.read_csv('MealNoMealData/mealData{}.csv'.format(i), header=None) for i in range(1, numFiles + 1)]
    noMeal = [pd.read_csv('MealNoMealData/Nomeal{}.csv'.format(i), header=None) for i in range(1, numFiles + 1)]

    for i in range(numFiles):
        meal[i] = transpose(meal[i])
        noMeal[i] = transpose(noMeal[i])

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

    if int(input('Pass From PCA? 1: YES, 0: NO:\t')) == 1:
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
    print("Support Vector Machine:\n\t Accuracy: {}\n\t 10-Fold CV:{}".format(round(max(svc_scores), 2), svc_scores))
    print("K Nearest Neighbor:\n\t Accuracy: {}\n\t 10-Fold CV:{}".format(round(max(knn_scores), 2), knn_scores))
    print("Logistic Regression:\n\t Accuracy: {}\n\t 10-Fold CV:{}".format(round(max(log_scores), 2), log_scores))
    print("Random Forest:\n\t Accuracy: {}\n\t 10-Fold CV:{}".format(round(max(rfc_scores), 2), rfc_scores))
    print("GaussianNB:\n\t Accuracy: {}\n\t 10-Fold CV:{}".format(round(max(gnb_scores), 2), gnb_scores))
    print()
