import numpy as np

class NNfunc(object):

    def rand(self, n):
        return np.random.normal(0,0.1,n)

    def relu(self, u):
        return np.array([max(0,x) for x in u])

    def d_relu(self, u):
        return np.array([1 if x>0 else 0 for x in u])

    def softmax(self, u):
        u = np.array(u) - np.max(u) # prevent overflow
        return np.array([np.exp(x) for x in u] / sum([np.exp(x) for x in u]))

    def dropout(self, size, p):
        bernoullis = np.random.binomial(1, p, size)
        return bernoullis
