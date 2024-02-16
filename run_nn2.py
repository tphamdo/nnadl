import load_mnist
import nn2 as _nn 

def main():
    train_data, validation_data, test_data = load_mnist.load_data_wrapper()

    nn = _nn.NeuralNet([784,120,10])
    nn.SGD(training_data=train_data, batch_size=10, epochs=60, eta=0.4, lmbda=4.0, evaluation_data=validation_data)
        
main()



