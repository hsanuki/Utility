# coding: utf-8
"""
ソートしてインデックスの修正
"""
train_coupon_df = train_coupon_df.sort(columns=["DISFROM"]).reset_index(drop=True)

""" 
pickleの読み込み書き込み方法
"""
pd.read_pickle, A.to_pickle()

"""
Seiresに対してapplymapするときはなるべくmapを使う
"""
dfTrain["qid"] = map(lambda q:  qid_dict[q],  dfTrain["query"])

"""
長さの異なるデータ同士の保存方法
"""
users.append({"user": user_vec[i],
              "coupon_ids": row_ids,
              "valid_coupon_ids": valid_row_ids})
"""
ixはインデックス，ilocは行番号を指定する
"""
df.ix[colA]
df.iloc[1]

"""
インデックスでmergeする
"""
pd.merge(left1, left2,  left_on=“key”, right_index=True)

"""
DataFrameの追加
"""
A = A.append(B, ignore_index=True)

"""
行持ち（いつもの）を列持ちに変換する
"""
df_store_sale = pd.pivot_table(df_store_sale, values="quantity", index=["area", "location", "natural_lawson_store", "pid", "date"], columns="segment")
df_store_sale.reset_index(inplace=True)

"""
別DataFrameから別DataFrameに変換する場合はcopyをつける
"""
df_train = df[df["y"].notnull()].copy()

"""
新たに評価関数を作成する
"""
from sklearn.metrics import make_scorer
def rmsle(predicted, actual, size):
    return np.sqrt(np.nansum(np.square(np.log(predicted + 1) - np.log(actual + 1)))/float(size))
scorer = make_scorer(rmsle, greater_is_better=False, size=10)
grid = GridSearchCV(est, param_grid, scoring=scorer)
"""
row - rowで欠損値を補完する
"""

In [129]: A
Out[129]:
     0    1  2    3
0  3.0  1.0  3  NaN
1  NaN  5.0  2  3.0
2  5.0  NaN  5  1.0
3  NaN  4.0  3  2.0

In [130]: A.apply(lambda x: x.fillna(value=A[2]))
Out[130]:
     0    1  2    3
0  3.0  1.0  3  3.0
1  2.0  5.0  2  3.0
2  5.0  5.0  5  1.0
3  3.0  4.0  3  2.0

"""
書き換えでWarning表示をなくす
"""
pd.options.mode.chained_assignment = None 

"""
階層型インデックス
"""
grouped.loc[:,(colA, colB)]
