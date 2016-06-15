import numpy as np

class NNfunc(object):

    def rand(self, size):
        return np.random.normal(0, 0.1, size)

    def relu(self, x):
        return np.clip(x, a_min=0, a_max=None)

    def d_relu(self, x):
        return np.where(x>0, 1, 0)

    def softmax(self, x):
        x -= np.max(x) # prevent overflow
        return np.exp(x) / np.sum(np.exp(x))

    def dropout(self, size, p):
        bernoullis = np.random.binomial(1, p, size)
        return bernoullis

nnfunc = NNfunc()
