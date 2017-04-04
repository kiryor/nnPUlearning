# Chainer example of PU learning and NNPU learning
This is a reproducing code for PU learning [1] and NNPU learning [2].

* ```pu_loss.py``` has a chainer implementation of the risk estimator for PU learning and NNPU learning. 
* ```train_mnist.py``` is an example code of PU learning and NNPU learning. 
The dataset is MNIST [3] preprocessed in such a way that even digits form the P class and odd digits form the N class.


## Requirements
* Python 3.6
* Numpy 1.1
* Chainer 1.22
* Scikit-learn 0.18

## Quick start
You can run an example code of MNIST for comparing the performance of PU learning and NNPU learning.

    python3.6 train_mnist.py -g 0

## Example result
After running ```training_mnist.py```, 2 figures and 1 log file are made in ```result/```
* Training error in ```result/training_error.png```

![training error](result/training_error.png "training error")

* Test error in ```result/test_error.png```

![test error](result/test_error.png "test error")


## Reference
[1] du Plessis, Marthinus Christoffel, Gang Niu, and Masashi Sugiyama. 
"Convex formulation for learning from positive and unlabeled data." 
Proceedings of The 32nd International Conference on Machine Learning. 2015.

[2] Kiryo, Ryuichi, Gang Niu, Plessis, Marthinus Christoffel, and Masashi Sugiyama. 
"Positive-Unlabeled Learning with Non-Negative Risk Estimator." arXiv preprint arXiv:1703.00593 (2017).
https://arxiv.org/abs/1703.00593

[3] http://yann.lecun.com/exdb/mnist/