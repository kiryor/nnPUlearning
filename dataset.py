import numpy as np
import urllib.request
import os
import tarfile
import pickle
from sklearn.datasets import fetch_openml

def get_mnist():
    mnist = fetch_openml('mnist_784', data_home=".")

    x = mnist.data
    y = mnist.target
    # reshape to (#data, #channel, width, height)
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)
    return (x_tr, y_tr), (x_te, y_te)


def binarize_mnist_class(y_train, y_test):
    y_train_bin = np.ones(len(y_train), dtype=np.int32)
    y_train_bin[y_train % 2 == 1] = -1
    y_test_bin = np.ones(len(y_test), dtype=np.int32)
    y_test_bin[y_test % 2 == 1] = -1
    return y_train_bin, y_test_bin


def unpickle(file):
    fo = open(file, 'rb')
    dictionary = pickle.load(fo, encoding='latin1')
    fo.close()
    return dictionary


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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, path=path)
        urllib.request.urlcleanup()

    x_tr = np.empty((0, 32 * 32 * 3))
    y_tr = np.empty(1)
    for i in range(1, 6):
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


def binarize_cifar10_class(y_train, y_test):
    y_train_bin = np.ones(len(y_train), dtype=np.int32)
    y_train_bin[(y_train == 2) | (y_train == 3) | (y_train == 4) | (y_train == 5) | (y_train == 6) | (y_train == 7)] = -1
    y_test_bin = np.ones(len(y_test), dtype=np.int32)
    y_test_bin[(y_test == 2) | (y_test == 3) | (y_test == 4) | (y_test == 5) | (y_test == 6) | (y_test == 7)] = -1
    return y_train_bin, y_test_bin


def make_dataset(dataset, n_labeled, n_unlabeled):
    def make_pu_dataset_from_binary_dataset(x, y, labeled=n_labeled, unlabeled=n_unlabeled):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        assert(len(x) == len(y))
        perm = np.random.permutation(len(y))
        x, y = x[perm], y[perm]
        n_p = (y == positive).sum()
        n_lp = labeled
        n_n = (y == negative).sum()
        n_u = unlabeled
        if labeled + unlabeled == len(x):
            n_up = n_p - n_lp
        elif unlabeled == len(x):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        _prior = float(n_up) / float(n_u)
        xlp = x[y == positive][:n_lp]
        xup = np.concatenate((x[y == positive][n_lp:], xlp), axis=0)[:n_up]
        xun = x[y == negative]
        x = np.asarray(np.concatenate((xlp, xup, xun), axis=0), dtype=np.float32)
        print(x.shape)
        y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))), dtype=np.int32)
        perm = np.random.permutation(len(y))
        x, y = x[perm], y[perm]
        return x, y, _prior

    def make_pn_dataset_from_binary_dataset(x, y):
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

    (x_train, y_train), (x_test, y_test) = dataset
    x_train, y_train, prior = make_pu_dataset_from_binary_dataset(x_train, y_train)
    x_test, y_test = make_pn_dataset_from_binary_dataset(x_test, y_test)
    print("training:{}".format(x_train.shape))
    print("test:{}".format(x_test.shape))
    return list(zip(x_train, y_train)), list(zip(x_test, y_test)), prior


def load_dataset(dataset_name, n_labeled, n_unlabeled):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = get_mnist()
        y_train, y_test = binarize_mnist_class(y_train, y_test)
    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = get_cifar10()
        y_train, y_test = binarize_cifar10_class(y_train, y_test)
    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))
    xy_train, xy_test, prior = make_dataset(((x_train, y_train), (x_test, y_test)), n_labeled, n_unlabeled)
    return xy_train, xy_test, prior
