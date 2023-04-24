import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


DATA_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data successfully loaded!")

    return X, y

def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATA_PATH)

    # create train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create the train/validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3d array -> (130, 13, 1) 130 = num_samples, 13 = num_coefficients, 1 = num_channels
    X_train = [..., np.newaxis]  # add an axis to array 4d array -> (num_samples, 130, 13, 1)
    X_validation = [..., np.newaxis]  # add an axis to array
    X_test = [..., np.newaxis]  # add an axis to array
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())


    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu')) # this layer is fully connected to the previous layer it is aimed to flatten the output of the previous layer
    model.add(keras.layers.Dropout(0.3))


    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y):
    
    X = X[np.newaxis, ...]

    # prediction = [[0.1, 0.2, ...]] -> 10 values represent the different classes genre
    prediction = model.predict(X) # X -> (1, 130, 13, 1)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1) # extract the index of the max value in the prediction array [4] will be the index of the max value, for example classical
    print("Expected index: {}, Predicted index: {}",format(y, predicted_index))


if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the CNN
    model.fit(X_train, y_train, validation_data=(X_validation,
                                                 y_validation), 
                                                 batch_size=32, 
                                                 epochs=30)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    
    # make predictions on a sample
    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)