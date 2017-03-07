# coding: utf-8
"""
否定形
"""
[~([True, False])]

"""
指定した文字列分の文字列の分割
"""
"AABBDD"[1:3]

"""
別ディレクトリのファイルのパスを通したい時
"""
# sys.path.insert(0, "/Users/sanukiharuyuki/Project/utils/")
# from utils import Df_process
from ..base import hogehogeclass
from .base import BaseEstimator

"""
None の判定
"""
if data is None

"""
辞書型のキーを決めずに宣言する
"""
import collections
collections.defaultdict(list)
"""
コマンドをPython側から実行する
"""
cmd = "python ./Programfile.py"
os.system(cmd)

"""
クラス変数・クラスメソッドの定義方法
"""
class Counter:
    count = 0
    @classmethod
    def next(cls):
        cls.count = cls.count + 1
        return cls.count

# クラス変数が必要ない場合はstaticmethodとして使う
    @staticmethod
    def test(a):
        print a

"""
  * , *args, **の意味
"""
#  * はunpackするという意味
def fun(a, b, c):
    print a, b, c
fun(*[a, b, c])

# *argsは任意文字列中のtupleを取得する
def func(*args):
    print args
func(2, 12, 4)
→ print 2, 12, 4
def func2(a, *args):
    print args
func(2, 12, 4)
→ print 12, 4

# **辞書型をunpackする
d = {"b": 5, "c": 7}
func(1, **d)

# **kwargsは先ほど*argsのdictバージョン
def func(a, **kwargs):
    print a, kwargs

"""
groupbyを用いたiteration
"""
In [43]: for name, group in df.groupby(['A', 'B']):
   ....:        print(name)
   ....:        print(group)
   ....: 
('bar', 'one')
     A    B         C         D
1  bar  one -0.042379 -0.089329
('bar', 'three')
     A      B        C         D
3  bar  three -0.00992 -0.945867

"""
文字列の変換
"""
str型→decode→unicode型
str型⇦encode⇦unicode型
UTF-8
shift-jis
 "\xe\xr4"     u"愛う"


"""
クラス定義の方法について
"""

"""
    型の例
    float
    int
    ~ object
    array_like, shape = [n_samples, n_features]
    DataFrame, shape = [n_samples, n_features]
    dict

    Returns
    ----------
    valueA: float

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.

    Attributes(内部で使われている変数の説明, 返り値としてクラスを返す場合にそのクラスの属性が何かを明示的にしめしたもの)
    ----------
    cv_: int
"""

"""
メソッドの継承
"""
class Superclass(object):
    @staticmethod
    def methodA():
        print "A"


class Subclass(Superclass):
    @staticmethod
    def methodB():
        print "B"



"""
global変数を関数内で使う
"""
var = 2
def f():
    global var
    print var

"""
使用メモリの表示
http: //www.sakito.com/2012/09/python-memoryprofiler.html
"""
@profile
def main():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    main()

python -m memory_profiler sample.py
"""
計測時間の表示
"""
import time

if __name__ == '__main__':
    start = time.time()
    for i in range(0,11):
        print "a"
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
"""
Python の変数は基本的に参照である
値渡しにしたければcopy()もしくはlistの場合は[:]スライスを使う
"""


"""
スクリプト内のみで使う関数の定義方法
"""
def _function(x):
    pass

"""
tupleで返り値を使用しない場合は_
"""
_, x = function_return_tuple()

"""
クラス名はキャメルケース（DataInitializer(名詞)), それ以外のメソッドはスネークケース（get_mean_value(動詞))
"""

"""
with構文
より安全・簡潔にその機能を使えるようにする構文. __enter__，__exit__がwithで定義したメソッドがあれば使える
"""
with open("msg.log", "w") as wfp:
    wfp.write("これはテストです")

