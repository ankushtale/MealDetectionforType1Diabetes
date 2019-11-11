from warnings import simplefilter
from train import TopFeatures, DropIfMaxNaN, handlingNaN, crosstab_stats
from features import Features
from classifiers import Classifiers

import pandas as pd
import numpy as np

simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    inputTest = pd.read_csv('{}.csv'.format(str(input('Enter Feature File name:\t'))), header=None)
    outputTest = pd.read_csv('{}.csv'.format(str(input('Enter Class File name:\t'))), header=None)

    outputTest.columns = ['Class']

    inputTest = inputTest.T
    inputTest = DropIfMaxNaN(inputTest)
    inputTest = handlingNaN(inputTest)

    Test_FeatureMatrix = pd.concat(
        [
            Features.Deviation(inputTest, 'N/A'),
            Features.meanRange(inputTest, 'N/A')[['MeanRange']],
            Features.Range(inputTest, 'N/A')[['HighRange', 'LowRange']],
            Features.FFT(inputTest, 'N/A')[['varFFT', 'sdFFT', 'meanFFT']],
            Features.Quantile(inputTest, 'N/A')['Quantile'],
        ],
        axis=1
    )

    if int(input('Pass From PCA? 1: YES, 0: NO:\t')) == 1:
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
        actual = np.array(outputTest['Class'])

        crosstab_stats(np.array(outputTest['Class']), pred)
