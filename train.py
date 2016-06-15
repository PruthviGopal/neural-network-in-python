import numpy as np
import nn
logfilename = 'NNH1'
nn.set_log(logfilename)

from sklearn.datasets import fetch_mldata
nn.logger.info('Downloading MNIST dataset.')
mnist = fetch_mldata('MNIST original')
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255
mnist.target = mnist.target.astype(np.int32)

train_img, test_img = np.split(mnist.data, [60000])
train_targets, test_targets = np.split(mnist.target, [60000])
train_data = list(zip(train_img, train_targets))
test_data = list(zip(test_img, test_targets))
np.random.shuffle(train_data)
np.random.shuffle(test_data)

import nnh1
net = nnh1.NNH1(n_units=[784,100,10])
net.train(train_data, test_data, batch_size=20, test_step=1000, epoch=10, \
            lr=0.01, lr_step=15000, lr_mult=0.1, wdecay=0.0005, momentum=0.9,\
            drop_ph=1.0, disp_step=100)
net.save('model.npz')
net.plot_log(logfilename+'.log')
