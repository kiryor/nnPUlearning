import numpy as np
import urllib.request
import os
import tarfile
import pickle
from sklearn.datasets import fetch_mldata


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


def binarize_mnist_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY % 2 == 1] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[_testY % 2 == 1] = -1
    return trainY, testY


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)


def get_cifar10(path="./mldata"):

    if not os.path.isdir(path):
        os.mkdir(path)
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = os.path.basename(url)
    full_path = os.path.join(path, file_name)
    folder = os.path.join(path, "cifar-10-batches-py")
    # if cifar-10-batches-py folder doesn't exists, download from website
    if not os.path.isdir(folder):
        print("download the dataset from {} to {}".format(url, path))
        urllib.request.urlretrieve(url, full_path)
        with tarfile.open(full_path) as f:
            f.extractall(path=path)
        urllib.request.urlcleanup()

    x_tr = np.empty((0,32*32*3))
    y_tr = np.empty(1)
    for i in range(1,6):
        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            x_tr = data_dict['data']
            y_tr = data_dict['labels']
        else:
            x_tr = np.vstack((x_tr, data_dict['data']))
            y_tr = np.hstack((y_tr, data_dict['labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    x_te = data_dict['data']
    y_te = np.array(data_dict['labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    # label_names = bm['label_names']
    # rehape to (#data, #channel, width, height)
    x_tr = np.reshape(x_tr, (np.shape(x_tr)[0], 3, 32, 32)).astype(np.float32)
    x_te = np.reshape(x_te, (np.shape(x_te)[0], 3, 32, 32)).astype(np.float32)
    # normalize
    x_tr /= 255.
    x_te /= 255.
    return (x_tr, y_tr), (x_te, y_te)  # , label_names


def binarize_cifar10_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[(_trainY==2)|(_trainY==3)|(_trainY==4)|(_trainY==5)|(_trainY==6)|(_trainY==7)] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[(_testY==2)|(_testY==3)|(_testY==4)|(_testY==5)|(_testY==6)|(_testY==7)] = -1
    return trainY, testY


def make_dataset(dataset, n_labeled, n_unlabeled):
    def make_PU_dataset_from_binary_dataset(x, y, labeled=n_labeled, unlabeled=n_unlabeled):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        assert(len(X) == len(Y))
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        n_p = (Y == positive).sum()
        n_lp = labeled
        n_n = (Y == negative).sum()
        n_u = unlabeled
        if labeled + unlabeled == len(X):
            n_up = n_p - n_lp
        elif unlabeled == len(X):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        prior = float(n_up) / float(n_u)
        Xlp = X[Y == positive][:n_lp]
        Xup = np.concatenate((X[Y == positive][n_lp:], Xlp), axis=0)[:n_up]
        Xun = X[Y == negative]
        X = np.asarray(np.concatenate((Xlp, Xup, Xun), axis=0), dtype=np.float32)
        print(X.shape)
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


def load_dataset(dataset_name, n_labeled, n_unlabeled):
    if dataset_name == "mnist":
        (trainX, trainY), (testX, testY) = get_mnist()
        trainY, testY = binarize_mnist_class(trainY, testY)
    elif dataset_name == "cifar10":
        (trainX, trainY), (testX, testY) = get_cifar10()
        trainY, testY = binarize_cifar10_class(trainY, testY)
    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))
    XYtrain, XYtest, prior = make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled)
    return XYtrain, XYtest, prior
