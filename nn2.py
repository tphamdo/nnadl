"""
An improvement on nn1.py
1. Uses cross entry cost function instead of MSE
2. Uses L2 regularization
3. Uses better weight initialization technique
"""

import random
import numpy as np
import load_mnist

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y, z=None):
        return (a-y)

class MSECost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(a, y, z):
        return (a-y) * sigmoid_prime(z)

class DefaultWeightInitializer(object):
    @staticmethod
    def biases(sizes):
        return [np.random.randn(y,1) for y in sizes[1:]]

    @staticmethod
    def weights(sizes):
        return [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1], sizes[1:])]

class LargeWeightInitializer(object):
    @staticmethod
    def biases(sizes):
        return [np.random.randn(y,1) for y in sizes[1:]]

    @staticmethod
    def weights(sizes):
        return [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

class NeuralNet:

    def __init__(self, sizes, cost=CrossEntropyCost, weightInitializer=DefaultWeightInitializer):
        self.biases = weightInitializer.biases(sizes)
        self.weights = weightInitializer.weights(sizes)
        self.cost = cost

    def SGD(self, training_data, batch_size, epochs, eta, lmbda=0, evaluation_data = None):
        """
        training_data = list of (x,y) pairs representing inputs and outputs to the net
        eta = training rate
        lmbda = regularization parameter
        """
        print("Doing SDG...")

        training_data = list(training_data)
        n_data = len(training_data)

        if evaluation_data: 
            evaluation_data = list(evaluation_data)
            n_test = len(evaluation_data)

        for i in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+batch_size] for k in range(0,n_data,batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n_data)

            print('Epoch {0} complete'.format(i+1))
            if evaluation_data: 
                print('Accuracy on evaluation data: {0}/{1}'.format(self.evaluate(evaluation_data), len(evaluation_data)))

            print('Accuracy on training data: {0}/{1}'.format(self.evaluate(training_data, is_training_data=True), len(training_data)))
            print('Cost on training data: {0}'.format(self.total_cost(training_data, lmbda)))
            print()

        pass


    def update_mini_batch(self, mini_batch, eta, lmbda, n_data):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [ nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [ nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    
        self.biases = [b-(eta/len(mini_batch)*nb) for b,nb in zip(self.biases, nabla_b)]
        self.weights = [(1-(eta*lmbda/n_data))*w-(eta/len(mini_batch)*nw) for w,nw in zip(self.weights, nabla_w)]

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

        # backprop
        delta = self.cost.delta(activations[-1],y)
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(len(self.weights)-2,-1,-1):
            z = zs[l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[l+1].transpose(),delta) * sigmoid_prime(zs[l])

            # update delta nabla b and w
            delta_nabla_b[l] = delta
            delta_nabla_w[l] = np.dot(delta, activations[l].transpose())

        return delta_nabla_b, delta_nabla_w

    def evaluate(self, data, is_training_data=False):
        if is_training_data:
            results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda):
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def feed_forward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

if __name__ == "__main__":
    train_data, validation_data, test_data = load_mnist.load_data_wrapper()

    nn = NeuralNet([784,120,10])
    nn.SGD(training_data=train_data, batch_size=10, epochs=60, eta=0.4, lmbda=4.0, evaluation_data=validation_data)
