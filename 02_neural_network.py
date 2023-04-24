import numpy as np

class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        # For example, layers = [784, 256, 10] would specify a neural network with an input layer of 784 neurons, a hidden layer of 256 neurons, and an output layer of 10 neurons.


        # Initialize random weights

        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

    
    def forward_propagate(self, inputs):
        
        activations = inputs

        for w in self.weights:

            # calculate the net inputs
            net_inputs = np.dot(activations, w)
            
            # calculate the activations
            activations = self._sigmoid(net_inputs)

            return activations
    
    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':

    # create a Multi-Layer Perceptron
    mlp = MLP()

    # create som inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perdorm forward propagation
    outputs = mlp.forward_propagate(inputs)

    # print the results
    print('The network input is: {}'.format(inputs))
    print('\n')
    print('The network output is: {}'.format(outputs))