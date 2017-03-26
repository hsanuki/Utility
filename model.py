# coding: utf-8
import chainer.functions as F
import chainer.links as L
from chainer import link, Chain, optimizers, Variable


class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.ConVolution2D(in_channel=50, out_channels=400, ksize=(3, 1)),
            fc1=L.Linear(7200, 3600),
            fc2=L.Linear(3600, 900),
            fc3=L.Linear(900, 50),
        )
        # input length=20

    def __call__(self, x):
        h = self.conv1(x)
