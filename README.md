This is a test of my understanding of neural nets from [neuralnetsanddeeplearning.com](http://neuralnetworksanddeeplearning.com/index.html).
<br />
<br />

<strong>Implementating a Neural Network by hand</strong>

There are two different implementations written by hand, nn1.py and nn2.py. Both are MLPs that use Stochastic Gradient Descent by backpropagation. nn1.py uses sigmoid activations and a Mean-Squared Error cost function. nn2.py improves on nn1.py by using a Cross-Entropy cost function, L2 regularization, and smarter weight initialization. On the MNIST dataset, nn1.py get 95.5% accuracy and nn2.py gets 98.0% accuracy
<br />
<br />

<strong>Using Pytorch</strong>

nn3.py uses Pytorch to create a convolutional net. The final model uses the input 28x28 images with a 20x24x24 convolutional layer, a 20x12x12 max pool layer, another 40x8x8 convolutional layer, a 40x4x4 max pool layer, a fully connected 100 neuron layer, and a final 10 neuron layer. The model uses ReLU activations for the hidden layers and log softmax for the output layer with a negative log likelihood loss function. The model gets 99.3% accuracy on the MNIST dataset. A copy of nn3.py is viewable in a jupyter notebook [here](https://colab.research.google.com/drive/15ErrDgsHbql2n5OIasHE1NhkI-5XZ9Fc?usp=sharing) (Note: You'll want to set the runtime to use GPUs or set args['cuda']=False)
<br />
<br />

<strong>TODO</strong>
<ol>
<li>use a fully-matrix based approach instead of iterating through each training example in the mini_batch. This should realize a speed up by taking advantage of linear algebra computational optimizations</li>
</ol>
<br />

<strong>Notes/Random Thoughts</strong>:
<ol>
    <li>The Cost function is calculated at the last layer. Each weight/bias is updated by taking the derivative of the Cost function with respect to that weight/bias. It's called backpropagation because the derivatives of layer $l$ can be written as a function of the derivatives at layer $l+1$ and therefore the derivatives are calculated or "propagated backwards" starting from the last layer to the first. This propogation backwards is a natural result of the chain-rule.</li>
    <li>I honestly can not believe how good (l2) regularization is. It's such a simple idea (it's a very small code change) to combat overfitting and yet provides massive improvements to the network's accuracy.</li>
    <li>nn3.py learns really fast. It consistenly gets to ~98% accuracy after just 1 epoch.</li>
</ol
