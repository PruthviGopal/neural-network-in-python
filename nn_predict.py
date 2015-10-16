import numpy as np
from sklearn.datasets import fetch_mldata

# download mnist data to 'scikit_learn_data/'
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
net = nn.NNH1(filename='nn_model_h1.npz')
# import nnh2
# net = nnh2.NNH2(filename='NNH2.npz')
net.predict(test_data[0], top_k=5)
net.predict(test_data[1], top_k=5)
net.predict(test_data[2], top_k=5)
