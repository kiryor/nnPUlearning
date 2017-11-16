import six
import copy
import argparse
import chainer

try:
    from matplotlib import use
    use('Agg')
except ImportError:
    pass

from chainer import Variable, functions as F
from chainer.training import extensions
from model import LinearClassifier, ThreeLayerPerceptron, MultiLayerPerceptron, CNN
from pu_loss import PULoss
from dataset import load_dataset



def process_args():
    parser = argparse.ArgumentParser(
        description='non-negative / unbiased PU learning Chainer implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', '-b', type=int, default=10000,
                        help='Mini batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='Zero-origin GPU ID (negative value indicates CPU)')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['figure1', 'exp-mnist', 'exp-cifar'],
                        help="Preset of configuration\n"+
                             "figure1: The setting of Figure1\n"+
                             "exp-mnist: The setting of MNIST experiment in Experiment\n"+
                             "exp-cifar: The setting of CIFAR10 experiment in Experiment")
    parser.add_argument('--dataset', '-d', default='mnist', type=str, choices=['mnist', 'cifar10'],
                        help='The dataset name')
    parser.add_argument('--labeled', '-l', default=100, type=int,
                        help='# of labeled data')
    parser.add_argument('--unlabeled', '-u', default=59900, type=int,
                        help='# of unlabeled data')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='# of epochs to learn')
    parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of nnPU')
    parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of nnPU')
    parser.add_argument('--loss', type=str, default="sigmoid", choices=['logistic', 'sigmoid'],
                        help='The name of a loss function')
    parser.add_argument('--model', '-m', default='3lp', choices=['linear', '3lp', 'mlp'],
                        help='The name of a classification model')
    parser.add_argument('--stepsize', '-s', default=1e-3, type=float,
                        help='Stepsize of gradient method')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()
    if args.gpu >= 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device_from_id(args.gpu).use()
    if args.preset == "figure1":
        args.labeled = 100
        args.unlabeled = 59900
        args.dataset = "mnist"
        args.batchsize = 30000
        args.model = "3lp"
    elif args.preset == "exp-mnist":
        args.labeled = 1000
        args.unlabeled = 60000
        args.dataset = "mnist"
        args.batchsize = 30000
        args.model = "mlp"
    elif args.preset == "exp-cifar":
        args.labeled = 1000
        args.unlabeled = 50000
        args.dataset = "cifar10"
        args.batchsize = 500
        args.model = "cnn"
        args.stepsize = 1e-5
    assert (args.batchsize > 0)
    assert (args.epoch > 0)
    assert (0 < args.labeled < 30000)
    if args.dataset == "mnist":
        assert (0 < args.unlabeled <= 60000)
    else:
        assert (0 < args.unlabeled <= 50000)
    assert (0. <= args.beta)
    assert (0. <= args.gamma <= 1.)
    return args


def select_loss(loss_name):
    losses = {"logistic": lambda x: F.softplus(-x), "sigmoid": lambda x: F.sigmoid(-x)}
    return losses[loss_name]


def select_model(model_name):
    models = {"linear": LinearClassifier, "3lp": ThreeLayerPerceptron,
              "mlp": MultiLayerPerceptron, "cnn": CNN}
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
    XYtrain, XYtest, prior = load_dataset(args.dataset, args.labeled, args.unlabeled)
    dim = XYtrain[0][0].size // len(XYtrain[0][0])
    train_iter = chainer.iterators.SerialIterator(XYtrain, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(XYtest, args.batchsize, repeat=False, shuffle=False)

    # model setup
    loss_type = select_loss(args.loss)
    selected_model = select_model(args.model)
    model = selected_model(prior, dim)
    models = {"nnPU": copy.deepcopy(model), "uPU": copy.deepcopy(model)}
    loss_funcs = {"nnPU": PULoss(prior, loss=loss_type, nnPU=True, gamma=args.gamma, beta=args.beta),
                  "uPU": PULoss(prior, loss=loss_type, nnPU=False)}
    if args.gpu >= 0:
        for m in models.values():
            m.to_gpu(args.gpu)

    # trainer setup
    optimizers = {k: make_optimizer(v, args.stepsize) for k, v in models.items()}
    updater = MultiUpdater(train_iter, optimizers, models, device=args.gpu, loss_func=loss_funcs)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(MultiEvaluator(test_iter, models, device=args.gpu))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(
                ['epoch', 'nnPU/loss', 'test/nnPU/error', 'uPU/loss', 'test/uPU/error', 'elapsed_time']))
    if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['nnPU/loss', 'uPU/loss'], 'epoch', file_name=f'training_error.png'))
            trainer.extend(
                extensions.PlotReport(['test/nnPU/error', 'test/uPU/error'], 'epoch', file_name=f'test_error.png'))
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
