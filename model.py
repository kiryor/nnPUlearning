import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain
from chainer.backends import cuda


class MyClassifier(Chain):
    prior = 0

    def __call__(self, x, t, loss_func):
        self.clear()
        h = self.calculate(x)
        self.loss = loss_func(h, t)
        chainer.reporter.report({'loss': self.loss}, self)
        return self.loss

    def clear(self):
        self.loss = None

    def calculate(self, x):
        return None

    def call_reporter(self, dictionary):
        chainer.reporter.report(dictionary, self)

    def error(self, x, t):
        xp = cuda.get_array_module(x, False)
        size = len(t)
        with chainer.no_backprop_mode():
            with chainer.using_config("train", False):
                h = xp.reshape(xp.sign(self.calculate(x).data), size)
        if isinstance(h, chainer.Variable):
            h = h.data
        if isinstance(t, chainer.Variable):
            t = t.data
        result = (h != t).sum() / size
        chainer.reporter.report({'error': result}, self)
        return cuda.to_cpu(result) if xp != np else result

    def compute_prediction_summary(self, x, t):
        xp = cuda.get_array_module(x, False)
        if isinstance(t, chainer.Variable):
            t = t.data
        n_p = (t == 1).sum()
        n_n = (t == -1).sum()
        with chainer.no_backprop_mode():
            with chainer.using_config("train", False):
                h = xp.ravel(xp.sign(self.calculate(x).data))
        if isinstance(h, chainer.Variable):
            h = h.data
        t_p = ((h == 1) * (t == 1)).sum()
        t_n = ((h == -1) * (t == -1)).sum()
        f_p = n_n - t_n
        f_n = n_p - t_p
        return int(t_p), int(t_n), int(f_p), int(f_n)


class LinearClassifier(MyClassifier, Chain):
    def __init__(self, prior, dim):
        super(LinearClassifier, self).__init__(
            l=L.Linear(dim, 1)
        )
        self.prior = prior

    def calculate(self, x):
        h = self.l(x)
        return h


class ThreeLayerPerceptron(MyClassifier, Chain):
    def __init__(self, prior, dim):
        super(ThreeLayerPerceptron, self).__init__(l1=L.Linear(dim, 100),
                                                   l2=L.Linear(100, 1))
        self.af = F.relu
        self.prior = prior

    def calculate(self, x):
        h = self.l1(x)
        h = self.af(h)
        h = self.l2(h)
        return h


class MultiLayerPerceptron(MyClassifier, Chain):
    def __init__(self, prior, dim):
        super(MultiLayerPerceptron, self).__init__(l1=L.Linear(dim, 300, nobias=True),
                                                   b1=L.BatchNormalization(300),
                                                   l2=L.Linear(300, 300, nobias=True),
                                                   b2=L.BatchNormalization(300),
                                                   l3=L.Linear(300, 300, nobias=True),
                                                   b3=L.BatchNormalization(300),
                                                   l4=L.Linear(300, 300, nobias=True),
                                                   b4=L.BatchNormalization(300),
                                                   l5=L.Linear(300, 1))
        self.af = F.relu
        self.prior = prior

    def calculate(self, x):
        h = self.l1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.l2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.l3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.l4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.l5(h)
        return h


class CNN(MyClassifier, Chain):
    def __init__(self, prior, dim):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 96, 3, pad=1),
            conv2=L.Convolution2D(96, 96, 3, pad=1),
            conv3=L.Convolution2D(96, 96, 3, pad=1, stride=2),
            conv4=L.Convolution2D(96, 192, 3, pad=1),
            conv5=L.Convolution2D(192, 192, 3, pad=1),
            conv6=L.Convolution2D(192, 192, 3, pad=1, stride=2),
            conv7=L.Convolution2D(192, 192, 3, pad=1),
            conv8=L.Convolution2D(192, 192, 1),
            conv9=L.Convolution2D(192, 10, 1),
            b1=L.BatchNormalization(96),
            b2=L.BatchNormalization(96),
            b3=L.BatchNormalization(96),
            b4=L.BatchNormalization(192),
            b5=L.BatchNormalization(192),
            b6=L.BatchNormalization(192),
            b7=L.BatchNormalization(192),
            b8=L.BatchNormalization(192),
            b9=L.BatchNormalization(10),
            fc1=L.Linear(None, 1000),
            fc2=L.Linear(1000, 1000),
            fc3=L.Linear(1000, 1),
        )
        self.af = F.relu
        self.prior = prior

    def calculate(self, x):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h
