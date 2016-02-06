import numpy as np
import nnfunc
nnf = nnfunc.NNfunc()
import nnlayer as nnl
import nn

class NNH1(nn.NN):

    def __init__(self, n_units=[], filename=''):
        if filename:
            params = np.load(filename)
            self.l1 = nnl.NNLayer(w=params['w1'], b=params['b1'])
            self.l2 = nnl.NNLayer(w=params['w2'], b=params['b2'])
        else:
            self.l1 = nnl.NNLayer(n_in=n_units[0], n_unit=n_units[1])
            self.l2 = nnl.NNLayer(n_in=n_units[1], n_unit=n_units[2])
            nn.logger.info('Net: %s' % n_units)

    def forward(self, datum, train=False):
        self.inputs = datum[0]
        z1 = self.l1.forward(self.inputs, nnf.relu, train)
        z2 = self.l2.forward(z1, nnf.softmax, train)
        loss = -np.log(z2[datum[1]])
        return z2, loss

    def backward(self, outputs, target):
        targets = np.array([1 if target == i else 0 for i in range(10)])
        delta2 = outputs - targets
        grad2w = self.l2.grad(delta2, self.l1.z)
        delta1 = self.l2.backward(delta2, self.l1.u, nnf.d_relu)
        grad1w = self.l1.grad(delta1, self.inputs)
        return grad1w, delta1, grad2w, delta2

    def set_dropout(self, drop_pi, drop_ph):
        if not drop_pi == 1: self.l1.set_dropout(drop_pi)
        if not drop_ph == 1: self.l2.set_dropout(drop_ph)

    def train_batch(self, data, lr, wdecay, momentum, drop_pi, drop_ph):
        N = len(data)
        self.set_dropout(drop_pi, drop_ph)
        outputs, loss = self.forward(data[0], train=True)
        grads = self.backward(outputs, data[0][1])
        w1, b1, w2, b2 = grads[0], grads[1], grads[2], grads[3]
        for n in range(1,N):
            self.set_dropout(drop_pi, drop_ph)
            outputs, loss_n = self.forward(data[n], train=True)
            loss += loss_n
            grads = self.backward(outputs, data[n][1])
            w1 += grads[0]
            b1 += grads[1]
            w2 += grads[2]
            b2 += grads[3]
        self.l1.update_params(w1/N, b1/N, lr, wdecay, momentum)
        self.l2.update_params(w2/N, b2/N, lr, wdecay, momentum)
        return loss/N

    def save(self, filename):
        np.savez(filename, \
                    w1=self.l1.w, b1=self.l1.b, w2=self.l2.w, b2=self.l2.b)
