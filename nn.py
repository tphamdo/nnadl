"""
A handmade implementation of stochastic gradient descent for a feedforward neural network i.e. 
without using pytorch, tensorflow, etc.
"""

import random

import numpy as np

class NeuralNet:

    def __init__(self, sizes):
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def print(self):
        print(self.biases)
        print(self.weights)

    def SGD(self, training_data, batch_size, epochs, eta, test_data = None):
        """
        training_data = list of (x,y) pairs representing inputs and outputs to the net
        eta = training rate
        """
        print("Doing SDG...")

        training_data = list(training_data)
        n_data = len(training_data)

        if test_data: 
            test_data = list(test_data)
            n_test = len(test_data)

        for i in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+batch_size] for k in range(0,n_data,batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                #print("biases: ", self.biases)
                #print("weights: ", self.weights)

            if (test_data):
                num_correct = self.evaluate(test_data) 
                print('Epoch {0}: {1}/{2}'.format(i+1, num_correct, len(test_data)))
            else: 
                print('Epoch {0} complete'.format(i+1))
        pass


    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [ nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [ nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    
        self.biases = [b-(eta/len(mini_batch)*nb) for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/len(mini_batch)*nw) for w,nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feed forward
        a = np.array(x)
        activations = [a]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            zs.append(z)

            a = sigmoid(z)
            activations.append(a)
        #print("zs ", zs)

        # backprop
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1]) 
        #print("delta ", delta)
        #print("activations ", activations)
        #print("act[1] ", activations[1])
        #print("type(act[1]) ", type(activations[1]))
        #print("act[1].transpose() ", activations[1].transpose())
        #print("act[0] ", activations[0])
        #print("type(act[0]) ", type(activations[0]))
        #print("act[0].transpose() ", activations[0].transpose())
        #print("mult ", np.dot(delta, activations[-2].transpose()))
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(len(self.weights)-2,-1,-1):
            z = zs[l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[l+1].transpose(),delta) * sigmoid_prime(zs[l])

            # update delta nabla b and w
            delta_nabla_b[l] = delta
            #print("activ l ", activations[l])
            delta_nabla_w[l] = np.dot(delta, activations[l].transpose())

        return delta_nabla_b, delta_nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def feed_forward(self, a):
        for w,b in zip(self.weights, self.biases):
            #print("w ", w)
            #print("a ", a)
            #print("b ", b)
            #print("np.dot(w,a)", np.dot(w,a))
            #print("np.dot(w,a)+b", np.dot(w,a)+b)
            a = sigmoid(np.dot(w,a) + b)
        return a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """ deriv = -(1 / (1 + np.exp(-x))^2) * np.exp(-x) * -1
              = np.exp(-x) / (1 + np.exp(-x))^2
              = (1 + np.exp(-x) - 1) / (1 + np.exp(-x))^2
              = 1 / (1 + np.exp(-x) - 1 / (1 + np.exp(-x))^2
              = 1 / (1 + np.exp(-x) * (1 - 1 / (1 + np.exp(-x)))
              = sigmoid(x) * (1 - sigmoid(x))
    """
    return sigmoid(x) * (1 - sigmoid(x))

nn = NeuralNet([3,5,9,1,4,3,2])
#nn.print()
#nn.SGD(training_data=[([[1],[2],[3]],[[1],[2]]),([[1],[2],[3]],[[1],[2]]),([[5],[6],[7]],[[3],[4]]),([[7],[8],[9]],[[4],[5]]),([[9],[10],[11]],[[5],[6]])], batch_size=2, epochs=1, eta=0.3, test_data=([[1],[2],[3]],[[1],[2]]))
