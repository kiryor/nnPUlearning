import six
import copy
import argparse
import numpy as np
import chainer

try:
    from matplotlib import use
    use('Agg')
except ImportError:
    pass

from chainer import Variable, functions as F
from chainer.training import extensions
from sklearn.datasets import fetch_mldata
from model import LinearClassifier, MultiLayerPerceptron
from pu_loss import PULoss


def get_mnist():
    mnist = fetch_mldata('MNIST original', data_home=".")
    x = mnist.data
    y = mnist.target
    # reshape to (#data, #channel, width, height)
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)
    return (x_tr, y_tr), (x_te, y_te)


def binarize_10_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY % 2 == 1] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[_testY % 2 == 1] = -1
    return trainY, testY


def make_dataset(dataset, n_labeled):
    def make_PU_dataset_from_binary_dataset(x, y, labeled=n_labeled):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        assert(len(X) == len(Y))
        perm = np.random.permutation(len(Y))
        unlabeled = len(Y) - labeled
        X, Y = X[perm], Y[perm]
        n_p = (Y == positive).sum()
        n_lp = labeled
        n_n = (Y == negative).sum()
        n_up = n_p - n_lp
        n_un = n_n
        n_u = unlabeled
        prior = float(n_up) / float(n_u)
        Xlp = X[Y == positive][:n_lp]
        Xup = np.concatenate((X[Y == positive][n_lp:], Xlp))[:n_up]
        Xun = X[Y == negative][:n_un]
        X = np.asarray(np.concatenate((Xlp, Xup, Xun)), dtype=np.float32)
        Y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))), dtype=np.int32)
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y, prior

    def make_PN_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)
        Y = np.asarray(np.concatenate((np.ones(n_p), -np.ones(n_n))), dtype=np.int32)
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y

    (_trainX, _trainY), (_testX, _testY) = dataset
    trainX, trainY, prior = make_PU_dataset_from_binary_dataset(_trainX, _trainY)
    testX, testY = make_PN_dataset_from_binary_dataset(_testX, _testY)
    print("training:{}".format(trainX.shape))
    print("test:{}".format(testX.shape))
    return list(zip(trainX, trainY)), list(zip(testX, testY)), prior


def process_args():
    parser = argparse.ArgumentParser(
        description='PU learning and NNPU learning Chainer example: MNIST even v.s. odd',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='Mini batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='Zero-origin GPU ID (negative value indicates CPU)')
    parser.add_argument('--labeled', '-l', default=1000, type=int,
                        help='# of labeled data')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='# of epochs to learn')
    parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of NNPU')
    parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of NNPU')
    parser.add_argument('--loss', type=str, default="sigmoid", choices=['logistic', 'sigmoid'],
                        help='The name of a loss function')
    parser.add_argument('--model', '-m', default='mlp', choices=['linear', 'mlp'],
                        help='The name of a classification model')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()
    assert (args.batchsize > 0)
    assert (args.epoch > 0)
    assert (0. <= args.beta)
    assert (0. <= args.gamma <= 1.)
    if args.gpu >= 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device_from_id(args.gpu).use()
    return args


def select_loss(loss_name):
    losses = {"logistic": lambda x: F.softplus(-x), "sigmoid": lambda x: F.sigmoid(-x)}
    return losses[loss_name]


def select_model(model_name):
    models = {"linear": LinearClassifier, "mlp": MultiLayerPerceptron}
    return models[model_name]


def make_optimizer(model, stepsize):
    optimizer = chainer.optimizers.Adam(alpha=stepsize)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.005))
    return optimizer


class MultiUpdater(chainer.training.StandardUpdater):

    def __init__(self, iterator, optimizer, model, converter=chainer.dataset.convert.concat_examples,
                 device=None, loss_func=None):
        assert(isinstance(model, dict))
        self.model = model
        assert(isinstance(optimizer, dict))
        if loss_func is None:
            loss_func = {k: v.target for k, v in optimizer.items()}
        assert(isinstance(loss_func, dict))
        super(MultiUpdater, self).__init__(iterator, optimizer, converter, device, loss_func)

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizers = self.get_all_optimizers()
        models = self.model
        loss_funcs = self.loss_func
        if isinstance(in_arrays, tuple):
            x, t = tuple(Variable(x) for x in in_arrays)
            for key in optimizers:
                optimizers[key].update(models[key], x, t, loss_funcs[key])
        else:
            raise NotImplemented


class MultiEvaluator(chainer.training.extensions.Evaluator):
    default_name = 'test'

    def __init__(self, *args, **kwargs):
        super(MultiEvaluator, self).__init__(*args, **kwargs)

    def evaluate(self):
        iterator = self._iterators['main']
        targets = self.get_all_targets()

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = chainer.reporter.DictSummary()
        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                if isinstance(in_arrays, tuple):
                    in_vars = tuple(Variable(x)
                                    for x in in_arrays)
                    for k, target in targets.items():
                        target.error(*in_vars)
                elif isinstance(in_arrays, dict):
                    in_vars = {key: Variable(x)
                               for key, x in six.iteritems(in_arrays)}
                    for k, target in targets.items():
                        target.error(**in_vars)
                else:
                    in_vars = Variable(in_arrays)
                    for k, target in targets.items():
                        target.error(in_vars)
            summary.add(observation)

        return summary.compute_mean()


def main():
    args = process_args()

    # dataset setup
    (trainX, trainY), (testX, testY) = get_mnist()
    trainY, testY = binarize_10_class(trainY, testY)
    dim = 784
    XYtrain, XYtest, prior = make_dataset(((trainX, trainY), (testX, testY)), args.labeled)
    train_iter = chainer.iterators.SerialIterator(XYtrain, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(XYtest, args.batchsize, repeat=False, shuffle=False)

    # model setup
    loss_type = select_loss(args.loss)
    selected_model = select_model(args.model)
    model = selected_model(prior, dim)
    models = {"nnpu": copy.deepcopy(model), "pu": copy.deepcopy(model)}
    loss_funcs = {"nnpu": PULoss(prior, loss=loss_type, NNPU=True, gamma=args.gamma, beta=args.beta),
                  "pu": PULoss(prior, loss=loss_type, NNPU=False)}
    if args.gpu >= 0:
        for m in models.values():
            m.to_gpu()

    # trainer setup
    optimizers = {k: make_optimizer(v, 1e-3) for k, v in models.items()}
    updater = MultiUpdater(train_iter, optimizers, models, device=args.gpu, loss_func=loss_funcs)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(MultiEvaluator(test_iter, models, device=args.gpu))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(
                ['epoch', 'nnpu/loss', 'test/nnpu/error', 'pu/loss', 'test/pu/error', 'elapsed_time']))
    if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['nnpu/loss', 'pu/loss'], 'epoch', file_name=f'training_error.png'))
            trainer.extend(
                extensions.PlotReport(['test/nnpu/error', 'test/pu/error'], 'epoch', file_name=f'test_error.png'))
    print("prior: {}".format(prior))
    print("loss: {}".format(args.loss))
    print("batchsize: {}".format(args.batchsize))
    print("model: {}".format(selected_model))
    print("beta: {}".format(args.beta))
    print("gamma: {}".format(args.gamma))
    print("")

    # run training
    trainer.run()


if __name__ == '__main__':
    main()
