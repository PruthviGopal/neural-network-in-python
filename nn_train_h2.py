import numpy as np
from sklearn.datasets import fetch_mldata

# download mnist data to current/scikit_learn_data/
# may take a few minutes
mnist = fetch_mldata('MNIST original')
mnist.data = mnist.data.astype(np.float32) # 784x70,000
mnist.data /= 255 # convert to [0,1]
mnist.target = mnist.target.astype(np.int32) # 1x70,000

# data for training in [:60000], for validation in [60000:]
train_img, test_img = np.split(mnist.data, [60000])
train_targets, test_targets = np.split(mnist.target, [60000])
train_data = zip(train_img, train_targets)
test_data = zip(test_img, test_targets)
np.random.shuffle(train_data)
np.random.shuffle(test_data)

import nn
import nnh2
logfilename = 'NNH2'
nn.set_log(logfilename)
net = nnh2.NNH2(n_units=[784,100,100,10])
net.train(train_data, test_data, batch_size=100, test_step=100, epoch=10,\
            lr=0.01, lr_step=3000, lr_mult=0.1, wdecay=0.0005, momentum=0.9, \
            drop_ph=1.0, disp_step=10)
net.save(logfilename+'.npz')
net.plot_log(logfilename+'.log')
