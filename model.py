import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, cuda

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
