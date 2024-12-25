import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, layer_dims, activation_functions):

        self.layer_dims = layer_dims
        self.activation_functions = activation_functions
        self.parameters = {}
        self.cache = {}
        self.costs = []
        self.accuracies = []

        # Initialize parameters
        self.initialize_parameters()

    def initialize_parameters(self):

        L = len(self.layer_dims)

        for l in range(1, L):
            # He initialization for weights
            self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], 
                self.layer_dims[l-1]) * np.sqrt(2/self.layer_dims[l-1])
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return Z > 0
    
    def tanh(self, Z):
        return np.tanh(Z)
    
    def tanh_derivative(self, Z):
        return 1 - np.power(self.tanh(Z), 2)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -709, 709)))
    
    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def get_activation(self, activation_name):
        activations = {
            'relu': (self.relu, self.relu_derivative),
            'tanh': (self.tanh, self.tanh_derivative),
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'softmax': (self.softmax, None)
        }
        return activations.get(activation_name.lower())
    
    def forward_propagation(self, X):
        self.cache['A0'] = X
        L = len(self.layer_dims)

        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            A_prev = self.cache[f'A{l-1}']

            # Linear forward
            Z = np.dot(W, A_prev) + b
            self.cache[f'Z{l}'] = Z

            # activation forward
            activation_func, _ = self.get_activation(self.activation_functions[l-1])
            self.cache[f'A{l}'] = activation_func(Z)

        return self.cache[f'A{L-1}']
    
    def compute_loss(self, AL, Y):
        m = Y.shape[1]

        if Y.shape[0] == 1:
            Y_one_hot = np.zeros((self.layer_dims[-1], m))
            Y_one_hot[Y.squeeze().astype(int), np.arange(m)] = 1
            Y = Y_one_hot

        cost = -1/m * np.sum(Y * np.log(AL + 1e-15))
        return cost
    
    def backward_propagation(self, Y):
        m = Y.shape[1]
        L = len(self.layer_dims)

        if Y.shape[0] == 1:
            Y_one_hot = np.zeros((self.layer_dims[-1], m))
            Y_one_hot[Y.squeeze().astype(int), np.arange(m)] = 1
            Y = Y_one_hot

        derivatives = {}

        # Output layer
        AL = self.cache[f'A{L-1}']
        dAL = -(np.divide(Y, AL + 1e-15) - np.divide(1 - Y, 1 - AL + 1e-15))

        # Handle each layer
        for l in reversed(range(1, L)):
            Z = self.cache[f'Z{l}']
            A_prev = self.cache[f'A{l-1}']
            W = self.parameters[f'W{l}']

            if l == L - 1 and self.activation_functions[l-1].lower() == 'softmax':
                dZ = AL - Y
            else:
                _, activation_derivative = self.get_activation(self.activation_functions[l-1])
                dZ = dAL * activation_derivative(Z)
                
            # Computing derivatives
            dW = 1 / m * np.dot(dZ, A_prev.T)
            db = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAL = np.dot(W.T, dZ)

            # Store derivatives
            derivatives[f'dW{l}'] = dW
            derivatives[f'db{l}'] = db

        return derivatives
    
    def update_prameters(self, derivatives, learning_rate):
        L = len(self.layer_dims)
        for l in range(1, L):
            self.parameters[f'W{l}'] -= learning_rate * derivatives[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * derivatives[f'db{l}']

    def get_accuracy(self, predictions, Y):
        if Y.shape[0] == 1:
            return np.mean(predictions == Y.squeeze())
        
        return np.mean(predictions == np.argmax(Y, axis=0))
    
    def predict(self, X):
        AL = self.forward_propagation(X)
        predictions = np.argmax(AL, axis=0)
        return predictions
    
    def train(self, X, Y, learning_rate = 0.01, num_iterations = 3000, print_cost=True):
        for i in range(num_iterations):
            AL = self.forward_propagation(X)

            cost = self.compute_loss(AL, Y)
            self.costs.append(cost)

            derivatives = self.backward_propagation(Y)

            self.update_prameters(derivatives, learning_rate)

            predictions = self.predict(X)
            accuracy = self.get_accuracy(predictions, Y)
            self.accuracies.append(accuracy)

            if print_cost and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}, Accuracy = {accuracy:.4f}")

        return self.costs, self.accuracies
    

data = pd.read_csv('/home/mouli/Desktop/codes/ann/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0].reshape(1, -1)
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0].reshape(1, -1)
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

print("X_train shape:", X_train.shape)  # Should be (784, m_train)
print("Y_train shape:", Y_train.shape)  # Should be (1, m_train)

# Example usage
# Define network architecture
layer_dims = [784, 128, 64, 10]  # Input layer, 2 hidden layers, output layer
activation_functions = ['relu', 'relu', 'softmax']

# Create and train network
nn = NeuralNetwork(layer_dims, activation_functions)

costs, accuracies = nn.train(X_train, Y_train, 
                            learning_rate=0.01, 
                            num_iterations=200)

# Make predictions
predictions = nn.predict(X_dev)
print(nn.get_accuracy(predictions, Y_dev))