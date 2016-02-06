import numpy as np
import nnfunc
nnf = nnfunc.NNfunc()

class NNLayer(object):

    def __init__(self, n_in=0, n_unit=0, w=[], b=[]):
        self.dropout = False
        if n_in: # init params
            self.w = nnf.rand(n_in*n_unit).reshape((n_unit,n_in))
            self.b = np.zeros(n_unit)
            self.len_inputs = n_in
        else: # load params
            self.w, self.b = w, b
            self.len_inputs = len(self.w)
        self.dw = np.zeros([len(self.w),len(self.w[0])])
        self.db = np.zeros(len(self.b)) # for momentum

    def forward(self, inputs, actv_f, train=False):
        if self.dropout:
            self.u = np.dot(self.w, self.drop(inputs, train)) + self.b
        else:
            self.u = np.dot(self.w, inputs) + self.b
        self.z = actv_f(self.u)
        return self.z

    def backward(self, deltas, u_lower, d_actv):
        grad = np.dot(np.transpose(self.w), deltas) * d_actv(u_lower)
        if self.dropout: return self.drop(grad, train=True)
        else: return grad

    def grad(self, deltas, z_lower):
        if self.dropout: z_lower = self.drop(z_lower, train=True)
        return np.tile( \
                deltas, (len(z_lower),1)).T * np.tile(z_lower, (len(deltas),1))

    def update_params(self, gw, gb, lr, wdecay, momentum):
        dw = -lr*(gw + wdecay*self.w) + momentum*self.dw
        db = -lr*gb + momentum*self.db
        self.w, self.b = self.w + dw, self.b + db
        self.dw, self.db = dw, db # for momentum

    def set_dropout(self, p):
        self.dropout = True
        self.drop_p = p
        self.drop_fltrs = nnf.dropout(self.len_inputs, p)

    def drop(self, inputs, train):
        if train:
            inputs *= self.drop_fltrs
            inputs /= self.drop_p
        return inputs
