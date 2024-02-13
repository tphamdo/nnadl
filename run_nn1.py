import load_mnist
import nn2 as _nn 

def main():
    train_data, validation_data, test_data = load_mnist.load_data_wrapper()

    nn = _nn.NeuralNet([784,60,50,10])
    nn.SGD(training_data=train_data, batch_size=30, epochs=30, eta=3, test_data=test_data)
        
main()



