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
    shandler = lg.StreamHandler()
    shandler.setFormatter(lg.Formatter('%(asctime)s %(levelname)s %(message)s'))
    shandler.setLevel(lg.INFO)
    logger.addHandler(shandler)

class NN(object):

    def test(self, data):
        accuracy_cnt ,sum_loss = 0, 0
        for i in range(len(data)):
            outputs, loss = self.forward(data[i])
            if data[i][1] == outputs.argmax(): accuracy_cnt += 1
            sum_loss += loss
        return accuracy_cnt*1.0 / len(data), sum_loss / len(data)

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
        its = list(range(0, len(data)-batch_size+1, batch_size)) * epoch
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
        categories = np.arange(0, 10, 1)
        results = list(zip(outputs, categories))
        results.sort(reverse=True)
        for rank, (score, name) in enumerate(results[:top_k], start=1):
            print('#%d | %s | %6.3f%%' % (rank, name, score*100))
        self.plot_pred(datum, results[:top_k])

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
            grads_df = (loss1 - loss2) / (2*eps)
            if grads_df > eps and grads[0][i][j] > eps:
                print('grad w1[%d][%d]' % (i, j))
                print('DF: '+str(grads_df))
                print('BP: '+str(grads[0][i][j]))
        for i, b in enumerate(self.l1.b):
            self.l1.b[i] += eps
            outputs, loss1 = self.forward(datum, train=True)
            self.l1.b[i] += -2*eps
            outputs, loss2 = self.forward(datum, train=True)
            self.l1.b[i] += eps
            grads_df = (loss1 - loss2) / (2*eps)
            if grads_df > eps and grads[1][i] > eps:
                print('grad b1[%d]' % (i))
                print('DF: '+str((loss1 - loss2) / (2*eps)))
                print('BP: '+str(grads[1][i]))

    def plot_log(self, filename):
        losses, accrs, test_losses, its_loss, its_test = ([] for i in range(5))
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
        fig, ax1 = plt.subplots()
        ax1.plot(its_test, accrs, 'b-')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('accuracy(%)', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        plt.legend(['test accuracy'], bbox_to_anchor=(1,0.95))
        ax2 = ax1.twinx()
        ax2.plot(its_loss, losses, ls='-', lw=0.5, color='#ffaa00')
        ax2.plot(its_test, test_losses, 'r-')
        ax2.set_ylabel('loss', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        plt.legend(['train loss', 'test loss'], bbox_to_anchor=(1,0.86))
        plt.tight_layout()
        plt.show()
