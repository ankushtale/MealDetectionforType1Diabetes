from sklearn import datasets
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import SVC

import pickle


class Classifiers:
    @classmethod
    def SVM_Hyperparametrs1(cls, inputTrain, outputTrain):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
        grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=5)
        grid_search.fit(inputTrain, outputTrain)
        print(grid_search.best_params_)

    @classmethod
    def SVM_Hyperparametrs(cls, Input, Output):
        X = Input
        y = Output

        inputTrain, inputTest, outputTrain, outputTest = train_test_split(
            X, y, test_size=0.5, random_state=0)

        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(inputTrain, outputTrain)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = outputTest, clf.predict(inputTest)
            print(classification_report(y_true, y_pred))
            print()

    @classmethod
    def SVC(cls, X_train, Y_train):
        svc = SVC(C=10, kernel='rbf', gamma=0.0001)
        svc.fit(X_train, Y_train)

        cls.save(svc, 'SVC')
        return svc

    @classmethod
    def LOG(cls, X_train, Y_train):
        log = LogisticRegression()
        log.fit(X_train, Y_train)

        cls.save(log, 'LOG')
        return log

    @classmethod
    def KNN(cls, X_train, Y_train):
        knn = KNeighborsClassifier()
        knn.fit(X_train, Y_train)

        cls.save(knn, 'KNN')
        return knn

    @classmethod
    def RFC(cls, X_train, Y_train):
        rfc = RandomForestClassifier(n_estimators=40)
        rfc.fit(X_train, Y_train)

        cls.save(rfc, 'RFC')
        return rfc

    @classmethod
    def Gaussian(cls, X_train, Y_train):
        gnb = GaussianNB()
        gnb.fit(X_train, Y_train)

        cls.save(gnb, 'GNB')
        return gnb

    @classmethod
    def save(cls, model, name):
        pickle.dump(model, open('{}_Training.sav'.format(name), 'wb'))

    @classmethod
    def load(cls, name):
        model = pickle.load(open('{}_Training.sav'.format(name), 'rb'))
        return model
