import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255
mnist.target = mnist.target.astype(np.int32)

train_img, test_img = np.split(mnist.data, [60000])
train_targets, test_targets = np.split(mnist.target, [60000])
test_data = list(zip(test_img, test_targets))
np.random.shuffle(test_data)

import nnh1
net = nnh1.NNH1(filename='model.npz')
net.predict(test_data[0], top_k=5)
net.predict(test_data[1], top_k=5)
net.predict(test_data[2], top_k=5)
