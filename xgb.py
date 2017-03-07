# -*- coding: utf-8 -*-
from sklearn import datasets, cross_validation
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss, confusion_matrix
import xgboost
import logging


class XGB():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        # 何故かbinaryだとエラーがでるので要注意する
        self.params.update({'objective': "multi:softprob"})
        # self.params.update({'objective': "reg:linear", "booster": "gbtree"})
        # 多クラス分類の時は，multi:softprob．２値分類の時は，binary:logistic，回帰の場合は"reg/linear"．boosterは"gbtree"

    def fit(self, X_train, Y_train, num_boost_round=None):
        self.colname = X_train.columns.values
        X_train, Y_train = X_train.as_matrix(), Y_train.values.tolist()
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgboost.DMatrix(X_train, label=Y_train)
        self.clf = xgboost.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X, Y=None, proba=True):
        predicted_proba = self.predict_proba(X)
        if proba: return predicted_proba
        predicted = np.argmax(predicted_proba, axis=1)
        if Y is None:
            predicted = pd.DataFrame(predicted)
            return predicted
        else:
            Y = Y.values.tolist()
            logging.debug(confusion_matrix(Y, predicted))
            logging.debug("score: {0}".format(log_loss(Y, predicted_proba)))

    def predict_proba(self, X):
        X = X.as_matrix()
        d_X = xgboost.DMatrix(X)
        return self.clf.predict(d_X)

    # ここを自由に変更する
    # デフォルトのloss function
    def score(self, X, y):
        Y = self.predict_proba(X)
        return log_loss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self

    def show_importance(self):
        feature_importance = self.clf.get_fscore()
        importance = [value for key, value in feature_importance.items()]
        feature_importance = dict(zip(self.colname, importance))
        print sorted(feature_importance.items(), key=lambda x: x[1])[::-1]

if __name__ == "__main__":

    iris=datasets.load_iris()
    trainX=iris.data[0::2,:]
    trainX=pd.DataFrame(trainX)
    trainX.columns=iris.feature_names
    trainY=iris.target[0::2]
    trainY = pd.Series(trainY)
    # 奇数番目を検証用データ
    testX=iris.data[1::2,:]
    testX=pd.DataFrame(testX)
    testX.columns=iris.feature_names
    testY=iris.target[1::2]
    testY = pd.Series(testY)

    clf = XGB(
        eval_metric = 'auc',
        num_class = len(np.unique(trainY)),
        nthread = 4,
        eta = 0.1,
        num_boost_round = 80,
        max_depth = 12,
        subsample = 0.5,
        colsample_bytree = 1.0,
        silent = 1,
        )

    parameters = {
        'num_boost_round': [100, 250, 500],
        # 学習率は0.01~0.2
        'eta': [0.05, 0.1, 0.3],
        # 3-10
        'max_depth': [6, 9, 12],
        # nodeの重みが一定値以下だと分割これ以上分割しない
        'min_child_weight': [1, 2, 3], 
        # 抽出割合．
        'subsample': [0.9, 1.0],
        # 木の葉nodeをさらに分割するのに必要な値
        'colsample_bytree': [0.9, 1.0],
        # 正則化項も一応あるけど，．．
        'gamma': [0,0.2,0.4,0.6,0.8]
    }
    # チューニング手順
    # max_depth，min_child_weight，gamma，subsample
    cv = cross_validation.KFold(len(trainX), n_folds=3, shuffle=True, random_state=1)
    grid = GridSearchCV(clf, parameters, n_jobs=1, cv=cv)
    grid.fit(trainX, trainY)
    clf.set_params(**grid.best_params_)
    clf.fit(trainX, trainY)
    clf.show_importance()
    clf.predict(testX, testY)
