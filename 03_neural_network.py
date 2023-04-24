import numpy as np
from random import random


# save the activation and derivatives when we compute the gradient descent
# implement the backpropagation algorithm
# implement the gradient descent algorithm
# implement the training algorithm
# train our network with some dummy dataset
# make predictions with our trained network


class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        # For example, layers = [784, 256, 10] would specify a neural network with an input layer of 784 neurons, a hidden layer of 256 neurons, and an output layer of 10 neurons.

        # Initialize random weights
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations # each array in the list represent the activations for a given layer

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives # each array in the list represent the derivatives for a given layer

    
    def forward_propagate(self, inputs):
        
        activations = inputs
        self.activations[0] = activations

        for i, w in enumerate(self.weights):

            # calculate matrix multiplication between previous activation and weight
            net_inputs = np.dot(activations, w)
            
            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        
        # a_3 = s(h_3)
        # h_3 = a_2 * W_2
        
        return activations
    
    def back_propagate(self, error, verbose=False):

        # dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i
        
        # s'(h_[i+1]) = a_[i+1] * (1 - a_[i+1])
        #               s(h_[i+1]) * (1 - s(h_[i+1]))
        
        # s(h_[i+1]) = 1 / (1 + e^(-h_[i+1])
        #              a_[i+1]        

        # delta = (y -a_[i+1]) s'(h_[i+i])
        # dE/dW_[i-1] = (y -a_[i+1]) s'(h_[i+1])) W_i s'(h_i) a_[i-1]
        
        # from left to right in the network layers
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            # a_[i+1]
            
            delta = error * self._sigmoid_derivative(activations)
            # delta = (y - a_[i+1]) s'(h_[i+1])

            delta_reshaped = delta.reshape(delta.shape[0], -1).T # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])

            current_activations = self.activations[i] # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]]) vertical vector
            # a_i
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)


            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)

            error = np.dot(delta, self.weights[i].T) # (y -a_[i+1]) s'(h_[i+1])) W_i s'(h_i)

            if verbose:
                print('Derivatives for W{}: {}'.format(i, self.derivatives[i]))


        return error
    

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            
            # print('Original: W{} {}'.format(i, weights))
            
            derivatives = self.derivatives[i]
            
            weights += derivatives * learning_rate
            
            # print('Updated: W{} {}'.format(i, weights))



    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                # perdorm forward propagation
                output = self.forward_propagate(input)

                # compute the error
                error = target - output

                # back propagatation
                self.back_propagate(error)

                # gradient descent

                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output) # mean squared error | average of the squared errors

            
            # report the error for each epoch
            print('Error: {} at epoch {}'.format(sum_error / len(inputs), i))



    def _mse(self, target, output):
        return np.average((target - output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 -x)
        # s(h_[i+1]) * (1 - s(h_[i+1]))
    


    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':

    # create a dataset to train a network for the sum operation
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)]) # array of 1000 arrays with 2 random numbers between 0 and 0.5 | ([0.1, 0.2], [0.3, 0.4])
    
    targets = np.array([[i[0] + i[1]] for i in inputs]) # array of 1000 arrays with the sum of the 2 numbers in the input array | ([0.3], [0.7])


    # create a Multi-Layer Perceptron
    mlp = MLP(2, [5], 1) # 2 inputs, 1 hidden layer with 5 neurons, 1 output

    # train our mlp
    mlp.train(inputs, targets, 50, 0.1)

    # create a dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)
    
    print('\n')

    print('My network believes that {} + {} is equal to {}'.format(input[0], input[1], output[0]))