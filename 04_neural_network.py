import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf

# array([[0.1, 0.2], [0.2, 0.2]])
# array([[0.3], [0.4]])

def generate_dataset(num_samples, test_size):

    # build inputs/targets for sum operation: y[0][0] = x[0][0] + x[0][1]
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    # split dataset into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    # create a dataset with 2000 samples
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
    print('x_test: {}'.format(x_test))
    print('y_test: {}'.format(y_test))

    # build a model: 2 -> 5 -> 1 (2 inputs, 5 neurons in hidden layer, 1 output)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation='sigmoid'), # input layer with 2 inputs and 5 neurons | 2 -> 5
        tf.keras.layers.Dense(1, activation='sigmoid') # output layer with 1 neuron | 5 -> 1
    ])

    # compile a model
    optimazer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimazer, loss='MSE')


    # train model
    model.fit(x_train, y_train, epochs=100)


    # evaluate model
    print('\n')
    print('Model evaluation:')
    model.evaluate(x_test, y_test, verbose=1)


    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)



    print('\nSome Predictions:')
    for d, p in zip(data, predictions):
        print('{} + {} = {}'.format(d[0], d[1], p[0]))