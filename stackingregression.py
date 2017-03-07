# coding: utf-8
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn import grid_search, cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.pipeline import Pipeline


class StackingRegressor:
    def __init__(self, first_models, first_params, second_model, second_params, kfold=4, scoring="mean_squared_error"):
        """
        Parameter
        --------
        first_models: model list [KNN, SVM]
        first_params: params list [param1, param2]
        seconf_model: model ex) KNN
        """
        self.kfold = kfold
        self.model_n = len(first_models)
        self.first_models = first_models
        self.first_params = first_params
        self.second_model = second_model
        self.second_params = second_params
        self.scoring = scoring
        self.best_params_ = [0 for _ in range(self.model_n)]
        self.predicted = [0 for _ in range(self.model_n)]
        self.test_predicted = [0 for _ in range(self.model_n)]
    
    def parametersearch_stage1(self, X_train, Y_train):
        logging.debug("Grid Search....")
        self.cv = cross_validation.KFold(len(X_train), n_folds=self.kfold, shuffle=True, random_state=71)
        for i in range(self.model_n):
            # TODO
            pipe_clf = Pipeline([("scl", StandardScaler()), ("clf", self.first_models[i])])
            grid = grid_search.GridSearchCV(pipe_clf, self.parameters[i], self.cv=self.cv, n_jobs=-1, scoring=self.scoring)
            grid.fit(X_train, Y_train)
            print "clf: {0} , params: {2} , score:{3}".format(self.first_models[i], grid.best_params_, grid.best_score_)
            self.best_params_[i] = grid.best_params_

    def predict_stage1(self, X_train, Y_train, X_test, is_filter=False):
        for i in range(self.model_n):
            self.first_models[i].set_params(**self.best_params_[i])
            pipe_clf = Pipeline([("scl", StandardScaler()), ("clf", self.first_models[i])])
            self.predicted[i] = cross_validation.cross_val_predict(pipe_clf, X_train, Y_train, cv=self.cv)
            pipe_clf.fit(X_train, Y_train)
            self.test_predicted[i] = pipe_clf.predict(X_test)

            # exclude outlier data
            if is_filter:
                self.test_predicted[i][self.test_predicted[i]<0] = 0
                self.test_predicted[i][self.test_predicted[i] > Y_train.max()] = Y_train.max()

    def makedata_stage2(self, X_train, X_test):
        self.data_train2 = pd.DataFrame([])
        self.data_test2 = pd.DataFrame([])
        for i in range(self.model_n):
            self.data_train2["clf" + str(i)] = self.predicted[i]
            self.data_test2["clf" + str(i)] = self.test_predicted[i]
            self.data_train2 = pd.concat([self.data_train2, X_train], axis=1)
            self.data_test2 = pd.concat([self.data_test2, X_test], axis=1)

    def parametersearch_stage2(self, X_train, Y_train):
        self.second_model = second_model
        self.second_params = second_params
        grid = grid_search.GridSearchCV(self.second_model, self.second_params, cv=self.cv, scoring=self.scoring)
        grid.fit(self.data_train2, Y_train)
        print "Final params: {0} score:{1}".format(grid.best_params_, grid.best_score_)
        self.second_params = grid.best_params_
    
    def predict_stage2(self, Y_train):
        self.final_clf.set_params(**self.final_best_param)
        self.final_clf.fit(self.data_train2, Y_train)
        predicted = self.final_clf.predict(self.data_test2)
        return predicted
