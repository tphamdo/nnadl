Implementation of a Neural Network using Stochastic Gradient Descent. The network uses sigmoid activations and a Mean-Squared Error cost function.

The network gets 95% accuracy on the MNIST dataset.

```shell-session
$ python3 ./mnist_nn.py
Doing SDG...
Epoch 1: 8829/10000
Epoch 2: 9125/10000
Epoch 3: 9239/10000
Epoch 4: 9315/10000
Epoch 5: 9371/10000
Epoch 6: 9385/10000
Epoch 7: 9435/10000
Epoch 8: 9464/10000
Epoch 9: 9465/10000
Epoch 10: 9480/10000
Epoch 11: 9499/10000
Epoch 12: 9513/10000
Epoch 13: 9522/10000
Epoch 14: 9523/10000
Epoch 15: 9535/10000
Epoch 16: 9526/10000
Epoch 17: 9530/10000
Epoch 18: 9536/10000
Epoch 19: 9546/10000
Epoch 20: 9548/10000
Epoch 21: 9554/10000
Epoch 22: 9536/10000
Epoch 23: 9547/10000
Epoch 24: 9564/10000
Epoch 25: 9561/10000
Epoch 26: 9548/10000
Epoch 27: 9556/10000
Epoch 28: 9559/10000
Epoch 29: 9547/10000
Epoch 30: 9553/10000 
```

** This is a test of my understanding of neural nets from [neuralnetsanddeeplearning.com](http://neuralnetworksanddeeplearning.com/index.html). Each weight/bias is updated by taking the derivative of the Cost function with respect to that weight/bias. It's called backpropagation because the derivatives of layer $l$ can be written as a function of the derivatives at layer $l+1$ and therefore the derivatives are calculated or "propagated backwards". This is a natural result of the chain-rule.

TODO: use a fully-matrix based approach instead of iterating through each training example in the mini_batch. This should realize a speed up by taking advantage of linear algebra computational optimizations.
