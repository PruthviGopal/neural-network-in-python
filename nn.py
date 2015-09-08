import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
np.random.seed(0)

import logging as lg
logger = lg.getLogger('NN')
logger.setLevel(lg.INFO)

def set_log(logfilename):
    logger.handlers = []
    fhandler = lg.FileHandler(filename=logfilename+'.log', mode='w')
    fhandler.setFormatter(lg.Formatter('%(asctime)s %(levelname)s %(message)s'))
    fhandler.setLevel(lg.INFO)
    logger.addHandler(fhandler)


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

nnf = NNfunc()


class NNLayer(object):

    def __init__(self, n_in=0, n_unit=0, w=[], b=[]):
        self.drop = False
        if (not n_in) and n_unit: # input layer
            self.n_unit = n_unit
            return
        if n_in: # init params
            self.w = nnf.rand(n_in*n_unit).reshape((n_unit,n_in))
            self.b = np.zeros(n_unit)
            self.n_unit = n_unit
        else: # load params
            self.w, self.b = w, b
            self.n_unit = len(self.w)
        self.dw = np.zeros([len(self.w),len(self.w[0])])
        self.db = np.zeros(len(self.b)) # for momentum

    def input(self, inputs, train=False):
        if self.drop: self.z = self.dropout(self.z, train)
        else: self.z = inputs
        return self.z

    def forward(self, inputs, actv_f, train=False):
        if self.drop:
            self.u = self.dropout(np.dot(self.w, inputs), train) + self.b
        else:
            self.u = np.dot(self.w, inputs) + self.b
        self.z = actv_f(self.u)
        return self.z

    def backward(self, w_upper, deltas_upper, d_actv):
        deltas = np.dot(np.transpose(w_upper), deltas_upper) * d_actv(self.u)
        if self.drop: deltas = self.dropout(deltas, train=True)
        return deltas

    def update_params(self, gw, gb, lr, wdecay, momentum):
        dw = -lr*(gw + wdecay*self.w) + momentum*self.dw
        db = -lr*gb + momentum*self.db
        self.w, self.b = self.w + dw, self.b + db
        self.dw, self.db = dw, db # for momentum

    def set_dropout(self, p):
        self.drop = True
        self.drop_p = p
        self.drop_fltrs = nnf.dropout(self.n_unit, p)

    def dropout(self, inputs, train):
        if train: inputs *= self.drop_fltrs # train
        else: inputs *= self.drop_p # test
        return inputs

class NNH1(object):

    def __init__(self, n_units=[], filename=''):
        if filename:
            params = np.load(filename)
            self.l0 = NNLayer(n_in=0, n_unit=len(params['w1']))
            self.l1 = NNLayer(w=params['w1'], b=params['b1'])
            self.l2 = NNLayer(w=params['w2'], b=params['b2'])
        else:
            self.l0 = NNLayer(n_in=0, n_unit=n_units[0])
            self.l1 = NNLayer(n_in=n_units[0], n_unit=n_units[1])
            self.l2 = NNLayer(n_in=n_units[1], n_unit=n_units[2])
            logger.info('Net: %s' % n_units)

    def forward(self, datum, train=False):
        z0 = self.l0.input(datum[0], train)
        z1 = self.l1.forward(z0, nnf.relu, train)
        z2 = self.l2.forward(z1, nnf.softmax)
        loss = -np.log(z2[datum[1]])
        return z2, loss

    def grad(self, delta, z_lower):
        return np.tile(delta, (len(z_lower),1)).T \
                    * np.tile(z_lower, (len(delta),1))

    def backward(self, outputs, target):
        targets = np.array([1 if target == i else 0 for i in range(10)])
        delta2 = outputs - targets
        grad2w = self.grad(delta2, self.l1.z)
        delta1 = self.l1.backward(self.l2.w, delta2, nnf.d_relu)
        grad1w = self.grad(delta1, self.l0.z)
        return grad1w, delta1, grad2w, delta2

    def set_dropout(self, drop_pi, drop_ph):
        if not drop_pi == 1: self.l0.set_dropout(drop_pi)
        if not drop_ph == 1: self.l1.set_dropout(drop_ph)

    def test(self, data):
        accuracy_cnt ,sum_loss = 0, 0
        for i in range(len(data)):
            outputs, loss = self.forward(data[i])
            if data[i][1] == outputs.argmax(): accuracy_cnt += 1
            sum_loss += loss
        return accuracy_cnt*1.0 / len(data), sum_loss / len(data)

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

    def train(self, data, test_data, batch_size=100, test_step=100, epoch=1, \
                lr=0.1, lr_step=0, lr_mult=1.0, wdecay=0.0005, \
                momentum=0.9, drop_pi=1.0, drop_ph=1.0, disp_step=10, \
                resume_it=0):
        logger.info('Training %d data.' % len(data))
        logger.info('Epoch: %d' % epoch)
        logger.info('Batch size: %d' % batch_size)
        logger.info('Base learning rate: %f' % lr)
        logger.info('Lr drop multiplier: %f' % lr_mult)
        logger.info('Lr drop step: %d' % lr_step)
        logger.info('Weight decay: %f' % wdecay)
        logger.info('Momentum: %f' % momentum)
        logger.info('Dropout (Input layer): %f' % drop_pi)
        logger.info('Dropout (Hidden layer): %f' % drop_ph)
        accuracy, test_loss = self.test(test_data)
        logger.info('Iteration %d accuracy: %f' % (resume_it,accuracy))
        logger.info('Iteration %d loss(test): %f' % (resume_it,test_loss))
        logger.info('Iteration %d lr: %f' % (resume_it,lr))
        its = range(0, len(data)-batch_size+1, batch_size) * epoch
        for i, it in enumerate(its):
            i += resume_it
            if it == 0: np.random.shuffle(data)
            loss = self.train_batch(data[it:it+batch_size], \
                                    lr, wdecay, momentum, drop_pi, drop_ph)
            if i % disp_step == 0:
                logger.info('Iteration %d loss: %f' % (i,loss))
            if (i+1) % test_step == 0:
                accuracy, test_loss = self.test(test_data)
                logger.info('Iteration %d accuracy: %f' % (i+1,accuracy))
                logger.info('Iteration %d loss(test): %f' % (i+1,test_loss))
            if lr_step > 0:
                if (i+1) % lr_step == 0:
                    lr *= lr_mult
                    logger.info('Iteration %d lr: %f' % (i+1,lr))
        logger.info('Optimization done.')

    def save(self, filename):
        np.savez(filename, \
                    w1=self.l1.w, b1=self.l1.b, w2=self.l2.w, b2=self.l2.b)

    def plot_pred(self, datum, results):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(datum[0].reshape(28,28), cmap='gray', interpolation='none')
        plt.title(datum[1])
        plt.grid(False)
        plt.subplot(1, 2, 2)
        y = np.arange(len(results)+1, 1, -1)
        x = [output for (output, category) in results]
        ylabel = [category for (outout, category) in results]
        plt.barh(y, x, height=0.5, align='center')
        plt.yticks(y, ylabel)
        plt.xlabel('output')
        plt.show()

    def predict(self, datum, top_k=5):
        print('Label: %s' % datum[1]),
        outputs, loss = self.forward(datum)
        categories = np.array([0,1,2,3,4,5,6,7,8,9])
        results = zip(outputs, categories)
        results.sort(reverse=True)
        self.plot_pred(datum, results[:top_k])
        print 'predicted as:'
        for rank, (score, name) in enumerate(results[:top_k], start=1):
            print('#%d | %s | %.3f%%' % (rank, name, score*100))

    def check_grad(self, datum, drop_pi=1, drop_ph=1, eps=0.0001):
        self.set_dropout(drop_pi, drop_ph)
        outputs, loss = self.forward(datum, train=True)
        grads = self.backward(outputs, datum[1])
        for (i, j), w in np.ndenumerate(self.l1.w):
            self.l1.w[i][j] += eps
            outputs, loss1 = self.forward(datum, train=True)
            self.l1.w[i][j] += -2*eps
            outputs, loss2 = self.forward(datum, train=True)
            self.l1.w[i][j] += eps
            print('grad w1[%d][%d]' % (i, j)),
            print 'DF: '+str((loss1 - loss2) / (2*eps)),
            print 'BP: '+str(grads[0][i][j])
        for i, b in enumerate(self.l1.b):
            self.l1.b[i] += eps
            outputs, loss1 = self.forward(datum, train=True)
            self.l1.b[i] += -2*eps
            outputs, loss2 = self.forward(datum, train=True)
            self.l1.b[i] += eps
            print('grad b1[%d]' % i),
            print 'DF: '+str((loss1 - loss2) / (2*eps)),
            print 'BP: '+str(grads[1][i])

    def subplot_log(self, xs, ys, labels, titles):
        plt.figure()
        for i in range(len(xs)-1):
            plt.subplot(len(xs)-1, 1, i+1)
            plt.xlabel('iteration')
            plt.ylabel(labels[i])
            plt.title(titles[i])
            if i == 0:
                plt.plot(xs[i][0], ys[i][0], label='loss(train)')
                plt.plot(xs[i][1], ys[i][1], label='loss(test)')
                plt.legend()
            else:
                plt.plot(xs[i], ys[i])
        plt.tight_layout()
        plt.show()

    def plot_log(self, filename):
        losses, lrs, accrs, test_losses, \
            its_loss, its_lr, its_test = ([] for i in range(7))
        logdata = open(filename)
        for line in logdata:
            line = line.split()
            if line[3] != 'Iteration': continue
            if line[5] == 'loss:':
                losses.append(line[6])
                its_loss.append(line[4])
            elif line[5] == 'loss(test):':
                test_losses.append(line[6])
                its_test.append(line[4])
            elif line[5] == 'accuracy:':
                accrs.append(100*float(line[6]))
            elif line[5] == 'lr:':
                if lrs:
                    lrs.append(lrs[-1])
                    its_lr.append(int(line[4])-1)
                lrs.append(line[6])
                its_lr.append(line[4])
        lrs.append(lrs[-1])
        its_lr.append(its_loss[-1])
        self.subplot_log([[its_loss, its_test], its_test, its_lr], \
                     [[losses, test_losses], accrs, lrs], \
                     ['loss', 'accuracy(%)', 'lr'], \
                     ['Loss', 'Accuracy', 'Lerning Rate'])
