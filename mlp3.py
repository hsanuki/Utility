# coding:utf-8
from chainer import Variable, FunctionSet, optimizers
import chainer.functions as F
import numpy as np
import pandas as pd

# 3-layer MLP
# 出力が0-1か複数ラベルかで出力層の形は異なる
# シグモイド．多クラスの場合はソフトマックス
# 回帰の場合は線形予測
class MLP3(FunctionSet):
    def __init__(self, mlpparams):
        # 基本的に最適化はAdamでいいんじゃないかな
        self.optimizer = optimizers.Adam()
        self.mlpparams = mlpparams
        super(MLP3, self).__init__(
            l1=F.Linear(mlpparams["input"], mlpparams["h1"]),
            l2=F.Linear(mlpparams["h1"], mlpparams["h2"]),
            l3=F.Linear(mlpparams["h2"], mlpparams["output"]))
        self.optimizer.setup(self.collect_parameters())

    def forward_raw(self, x, train=True):
        h = x
        # reluかsigmoidかtanhをを使う
        h = F.dropout(F.relu(self.l1(h)), ratio=self.mlpparams["dropout1"], train=train)
        h = F.dropout(F.relu(self.l2(h)), ratio=self.mlpparams["dropout2"], train=train)
        # 回帰の場合は線形予測でやればいい．
        y = self.l3(h)
        return y

    def predict(self, x_data, y_data=None):
        x_data = x_data.astype(np.float32)
        if type(x_data)==pd.DataFrame: x_data = x_data.as_matrix()
        x = Variable(x_data)
        y = self.forward_raw(x, train=False)
        if y_data is None:
            # ２値データだからシグモイド関数を使ってる
            # return F.sigmoid(y).data.reshape(x_data.shape[0])
            # 回帰：return y.data
            return F.softmax(y).data
        else:
            if type(y_data)==pd.DataFrame: y_data = y_data.as_matrix()
            y_data = y_data.astype(np.int32)
            # return F.sigmoid_cross_entropy(y, Variable(y_data)).data
            # return F.mean_squared_error(y, Variable(y_data)).data
            return F.softmax_cross_entropy(y, Variable(y_data)).data

    def forward(self, x_data, y_data):
        x, t = Variable(x_data), Variable(y_data)
        y = self.forward_raw(x, train=True)
        # 回帰の場合はrmseでやる
        # 回帰：return F.mean_squared_error(y, t)
        return F.softmax_cross_entropy(y, t)
        # return F.sigmoid_cross_entropy(y, t)

    def learning_rate_decay(self, rate):
        self.optimizer.lr *= rate

    def train(self, x, y, n_epoch, evalset=None, batchsize=128, verbose=True):
        x = x.astype(np.float32)
        x = x.as_matrix()
        y = y.astype(np.int32)
        y = y.as_matrix()
        for epoch in xrange(1, n_epoch):
            print '**** epoch {0}/{1}'.format(epoch, n_epoch)
            if epoch == 100:
                self.learning_rate_decay(0.5)
            perm = np.random.permutation(x.shape[0])
            sum_loss = 0.0
            c = 0
            for i in xrange(0, x.shape[0], batchsize):
                x_batch = x[perm[i: i + batchsize]]
                y_batch = y[perm[i: i + batchsize]]
                self.optimizer.zero_grads()
                loss = self.forward(x_batch, y_batch)
                loss.backward()

                self.optimizer.update()
                loss = float(loss.data)
                sum_loss += loss * batchsize
                c += x_batch.shape[0]

            if verbose:
                print "train logloss: {}".format(sum_loss / c)
                if evalset is not None:
                    x, y = evalset[0], evalset[1]
                    x = x.astype(np.float32)
                    x = x.as_matrix()
                    y = y.astype(np.int32)
                    y = y.as_matrix()
                    perm = np.random.permutation(x.shape[0])
                    for i in xrange(0, x.shape[0], batchsize):
                        x_batch = x[perm[i: i + batchsize]]
                        y_batch = y[perm[i: i + batchsize]]
                        loss = self.predict(x_batch, y_batch)
                        sum_loss += loss * batchsize
                        c += x_batch.shape[0]
                    print "valid logloss: {}".format(sum_loss / c)
