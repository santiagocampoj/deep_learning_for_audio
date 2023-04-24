import math

def sigmoid(x):
    y = 1.0 / (1 + math.exp(-x))
    return y

def activate(inputs, weights):
    # perform net input
    h = 0
    for x, w in zip(inputs, weights):
        h += x * w

    # perform the activation
    return sigmoid(h)


if __name__ == '__main__':
    
    inputs = [1, 2, 3, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    output = activate(inputs, weights)

    print(output)
