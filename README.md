# Neural Network in Python
* 1 hidden layer (and 2 hidden layers) NN in Python 2
* Including a trained model of handwritten digit database MNIST

## Dependencies
Python 2, NumPy, scikit-learn, matplotlib  

## Examples
* A sample of '8' and its outputs of NN
![Prediction](/examples/NNH1_pred.png)  

* Loss, accuracy and training iteration  
![Log of training](/examples/NNH1_train_log.png)  

Reaches 98% accuracy after 60000 iterations (may take around 10 min.)
```py
import nn
net = nn.NNH1(n_units=[784,100,10])
net.train(train_data, test_data, batch_size=10, test_step=1000, epoch=10, \
            lr=0.01, lr_step=30000, lr_mult=0.1, wdecay=0.0005, momentum=0.9,\
            drop_ph=1.0, disp_step=100)
```
