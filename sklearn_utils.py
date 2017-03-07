# coding: utf-8
# Sklearnに関連するもの
"""
欠損値補完
Pipelineで繋ぎたいときとかに欠損値補完をfillnaではなく，Imputerでやる
"""
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])

"""
目的変数のクラスラベルのエンコーディングに使う
"""
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df[“classlabel”].values)
#  >>> y = array([0, 1, 2])

"""
パイプラインを用いた交差検定，学習曲線の可視化，グリッドサーチ
"""
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=10)
train_sizes, train_scores, test_scores = learning_curve(estimator = pipe_lr, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=10)
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=“accuracy”, cv=10, n_jobs=-1)

"""
複数学習器による多数決額数（クラス分類のAveraging）
"""
from sklearn.grid_search import GridSearchCV
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}
#  ここの書き方気をつける  →  ”名前__parametername"
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)

"""
時系列データのcross validation
"""
idxs = range(len(X_train))
cv = [(idxs[: len(X_train) / 10 * i], idxs[len(X_train) / 10 * i: len(X_train) / 10 * (i + 1)]) for i in np.arange(5, 10)]

"""
非線形なものにも利用可能な相関係数
"""
from minepy import MINE
m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x**2)
print m.mic()
"""
grid_scoreとcross_val_score
"""
両者はcvのデータ数が異なる場合は違う値をとる
"""
識別器のscorerについて
"""
# 何も指定しない場合は学習器のデフォルトのスコア関数（→Xgboost側で設定できる）
grid = GridSearchCV(self.clf, self.params, n_jobs=-1, cv=self.cv, verbose=verbose)
# 何も設定しない場合は回帰とかそれぞれ毎に決まっている（→ Xgboost側で設定できない）
scores = cross_val_score(self.clf, X_train, Y_train, cv=self.cv)
"""
パラメータのGrid作成
"""
from sklearn.grid_search import ParameterGrid
params = {"environment_weight": np.arange(0,1.1,0.1), "social_weight": np.arange(0, 1.1, 0.1), "governance_weight": np.arange(0, 1.1, 0.1)}
result = dict()
for param in ParameterGrid(params):
    if np.sum(param.values())==0: continue
    weight = [param["environment_weight"]] * len(environment_columns) + [param["governance_weight"]] * len(governance_columns) + [param["social_weight"]] * len(social_columns)
    score = calculate_score(df,weight)
    result[score] = param
best_index = np.argsort(result.keys())[-1]
best_param = result.values()[best_index]
best_score = result.keys()[best_index]

"""
LinearRegressionはnormalize=Trueになっていても，偏回帰係数の値は標準化される前のもの．
"""
