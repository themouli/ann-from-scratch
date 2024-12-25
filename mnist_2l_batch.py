import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/home/mouli/Desktop/codes/ann/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    return exp_Z / np.sum(exp_Z, axis = 0, keepdims = True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, m, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2




def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def get_batches(X, Y, batch_size):
    """
    Generate batches from input data X and Y

    Returns:
    (X_batch, Y_batch) tuples
    """

    m = X.shape[1]

    indices = np.random.permutation(m)

    for i in range(0, m, batch_size):
        batch_indices = indices[i:min(i+batch_size, m)]
        X_batch = X[:, batch_indices]
        Y_batch = Y[batch_indices]
        yield X_batch, Y_batch


def gradient_descent_with_batches(X, Y, alpha, iterations, batch_size):
    m = X.shape[1]
    W1, b1, W2, b2 = init_params()
    costs = []
    accuracies = []

    for i in range(iterations):
        epoch_cost = 0
        epoch_accuracy = 0
        num_batches = 0

        for X_batch, Y_batch in get_batches(X, Y, batch_size):
            batch_size_current = X_batch.shape[1]

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)

            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, batch_size_current, X_batch, Y_batch)

            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            predictions = get_predictions(A2)

            batch_accuracy = get_accuracy(predictions, Y_batch)

            epoch_accuracy += batch_accuracy

            num_batches += 1

        epoch_accuracy /= num_batches

        if i % 25 == 0:
            print(f"Iteration {i}, Accuracy: {epoch_accuracy:.4f}")
            accuracies.append(epoch_accuracy)

    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


W1, b1, W2, b2 = gradient_descent_with_batches(X_train, Y_train, 0.05, 500, m_train)


test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)