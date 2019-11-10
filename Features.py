from functools import reduce
import pandas as pd
import numpy as np
from scipy.fftpack import fft


class Features:
    @classmethod
    def Deviation(cls, df, result):
        x = pd.DataFrame(columns=['inRangeCount', 'LowCount', 'HighCount', 'LowMean', 'HighMean', 'Class'])
        for i in df.columns:
            inRange = list(filter(lambda x: 70 <= x <= 180, df[i]))
            High = list(filter(lambda x: x > 180, df[i]))
            Low = list(filter(lambda x: x < 70, df[i]))
            x = x.append(
                {
                    'inRangeCount': len(inRange),
                    'LowCount': len(Low),
                    'HighCount': len(High),
                    'LowMean': round(reduce(lambda x, y: x + y, Low) / len(Low), 2) if Low else 0,
                    'HighMean': round(reduce(lambda x, y: x + y, High) / len(High), 2) if High else 0,
                    'Class': result
                },
                ignore_index=True
            )
        return x

    @classmethod
    def meanRange(cls, df, result):
        range_list = []
        x = pd.DataFrame(columns=['MeanRange', 'Class'])
        for col in df.columns:
            max_val = df[col].max()
            min_val = df[col].min()
            range_list.append(max_val - min_val)

        mean = sum(range_list) / len(range_list)
        for i in range(len(range_list)):
            x = x.append(
                {
                    'MeanRange': round(range_list[i] - mean),
                    'Class': result
                },
                ignore_index=True
            )

        return x

    @classmethod
    def Range(cls, df, result):
        x = pd.DataFrame(columns=['HighRange', 'LowRange', 'Class'])
        for i in df.columns:
            x = x.append(
                {
                    'LowRange': LowRange(list(df[i]))[0],
                    'HighRange': HighRange(list(df[i]))[0],
                    'Class': result
                },
                ignore_index=True
            )
        return x

    @classmethod
    def FFT(cls, df, result):
        x = pd.DataFrame(columns=['varFFT', 'sdFFT', 'meanFFT', 'Class'])
        for i in df.columns:
            yf = np.abs(fft(df[i]))
            x = x.append(
                {
                    'varFFT': round(np.var(yf), 4),
                    'sdFFT': round(np.std(yf), 2),
                    'meanFFT': round(np.mean(yf), 2),
                    'Class': result
                },
                ignore_index=True
            )
        return x

    @classmethod
    def Quantile(cls, df, result):
        x = pd.DataFrame(columns=['Quantile', 'Class'])
        for i in df.columns:
            x = x.append(
                {
                    'Quantile': df[i].quantile(0.5),
                    'Class': result
                },
                ignore_index=True
            )
        return x


def LowRange(lst):
    Min = float('-inf')
    prev = lst[0]
    minlst = []
    currentLst = [lst[0]]
    current = 0
    for i in range(1, len(lst)):
        if prev >= lst[i]:
            current += prev - lst[i]
            currentLst.append(lst[i])
        if prev < lst[i]:
            if Min < current:
                Min = current
                minlst = currentLst[:]
            current = 0
            currentLst = [lst[i]]
        prev = lst[i]

    if currentLst:
        if Min < current:
            Min = current
            minlst = currentLst[:]
    return Min, minlst


def HighRange(lst):
    Max = float('-inf')
    prev = lst[0]
    maxlst = []
    currentLst = [lst[0]]
    current = 0
    for i in range(1, len(lst)):
        if prev <= lst[i]:
            current += lst[i] - prev
            currentLst.append(lst[i])
        if prev > lst[i]:
            if Max < current:
                Max = current
                maxlst = currentLst[:]
            current = 0
            currentLst = [lst[i]]
        prev = lst[i]

    if currentLst:
        if Max < current:
            Max = current
            maxlst = currentLst[:]
    return Max, maxlst
