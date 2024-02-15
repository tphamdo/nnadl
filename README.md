<h3>Implementating a Neural Network by hand.</h3>

There are two different implementations nn1.py and nn2.py. Both use Stochastic Gradient Descent. nn1.py uses sigmoid activations and a Mean-Squared Error cost function. nn2.py improves on nn1.py by using a Cross-Entropy cost function and L2 regularization. On the MNIST dataset, nn1.py get 95.5% accuracy and nn2.py gets 97.9% accuracy

```shell-session
$ python3 ./run_nn1.py
Doing SDG...
Epoch 1: 8829/10000
Epoch 2: 9125/10000
.
.
Epoch 29: 9553/10000 
Epoch 30: 9547/10000
```

```shell-session
$ python3 ./run_nn2.py
Doing SDG...
Epoch 1: 9299/10000
Epoch 2: 9448/10000
.
Epoch 59: 9786/10000
Epoch 60: 9789/10000
```

<strong>TODO:</strong>
1. use a fully-matrix based approach instead of iterating through each training example in the mini_batch. This should realize a speed up by taking advantage of linear algebra computational optimizations<br />
2. optimize nn2.py with weight initializer/regulation

<strong>Notes</strong>: This is a test of my understanding of neural nets from [neuralnetsanddeeplearning.com](http://neuralnetworksanddeeplearning.com/index.html). The Cost function is calculated at the last layer. Each weight/bias is updated by taking the derivative of the Cost function with respect to that weight/bias. It's called backpropagation because the derivatives of layer $l$ can be written as a function of the derivatives at layer $l+1$ and therefore the derivatives are calculated or "propagated backwards" starting from the last layer to the first. This propogation backwards is a natural result of the chain-rule.

