import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_dims, activation_functions):

        self.layer_dims = layer_dims
        self.activation_functions = activation_functions
        self.parameters = {}
        self.cache = {}
        self.costs = []
        self.accuracies = []
        self.val_accuracies = []

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
    
    def get_batches(self, X, Y, batch_size):
        m = X.shape[1]
        indices = np.random.permutation(m)

        for i in range(0, m, batch_size):
            batch_indices = indices[i:min(i+batch_size, m)]
            X_batch = X[:, batch_indices]
            Y_batch = Y[:, batch_indices] if len(Y.shape) > 1 else Y[batch_indices]
            yield X_batch, Y_batch

    def get_accuracy(self, predictions, Y):
        if Y.shape[0] == 1:
            return np.mean(predictions == Y.squeeze())
        
        return np.mean(predictions == np.argmax(Y, axis=0))
    
    def predict(self, X):
        AL = self.forward_propagation(X)
        predictions = np.argmax(AL, axis=0)
        return predictions
    
    def train(self, X, Y, X_dev, Y_dev, learning_rate = 0.01, num_epochs = 3000, batch_size = 32, print_cost=True):
        m = X.shape[1]

        for epoch in range(num_epochs):
            epoch_cost = 0
            epoch_accuracy = 0
            num_batches = 0

            for X_batch, Y_batch in self.get_batches(X, Y, batch_size):
                AL = self.forward_propagation(X_batch)

                batch_cost = self.compute_loss(AL, Y_batch)
                epoch_cost += batch_cost

                derivatives = self.backward_propagation(Y_batch)

                self.update_prameters(derivatives, learning_rate)

                predictions = self.predict(X_batch)
                batch_accuracy = self.get_accuracy(predictions, Y_batch)
                epoch_accuracy += batch_accuracy

                num_batches += 1

            epoch_cost /= num_batches
            epoch_accuracy /= num_batches

            # Validation phase
            dev_predictions = nn.predict(X_dev)
            dev_accuracy = self.get_accuracy(dev_predictions, Y_dev)

            self.costs.append(epoch_cost)
            self.accuracies.append(epoch_accuracy)
            self.val_accuracies.append(dev_accuracy)


            if print_cost and epoch % 50 == 0:
                print(f"Epoch {epoch}/{num_epochs}: "
                    f"Cost = {epoch_cost:.4f}, "
                    f"Train Accuracy = {epoch_accuracy:.4f}, "
                    f"Val Accuracy = {dev_accuracy:.4f}")
                
        return self.costs, self.accuracies, self.val_accuracies
    
    def get_parameter_count(self):
        total_params = 0
        param_dict = {}

        L = len(self.layer_dims)

        for l in range(1, L):
            W_params = self.layer_dims[l] * self.layer_dims[l-1]
            b_params = self.layer_dims[l]
            layer_params = W_params + b_params
            
            param_dict[f'layer_{l}'] = {
                'weights': W_params,
                'biases': b_params,
                'total': layer_params
            }

            total_params += layer_params

        return total_params, param_dict
    
    def print_model_summary(self):
        total_params, layer_params = self.get_parameter_count()
        print("\nNeural Network Architecture Summary:")
        print("====================================")
        print(f"Total layers: {len(self.layer_dims) - 1}")
        print("\nLayer Details:")
        print("-------------")
        
        for i, (neurons, activation) in enumerate(zip(self.layer_dims[1:], self.activation_functions)):
            layer_num = i + 1
            params = layer_params[f'layer_{layer_num}']
            print(f"\nLayer {layer_num}:")
            print(f"  Neurons: {neurons}")
            print(f"  Activation: {activation}")
            print(f"  Weights: {params['weights']:,}")
            print(f"  Biases: {params['biases']:,}")
            print(f"  Total Parameters: {params['total']:,}")
        
        print("\nTotal Network Parameters:", f"{total_params:,}")


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
layer_dims = [784, 128, 64, 10]  
activation_functions = ['relu', 'relu', 'softmax'] 

# Create and train network
nn = NeuralNetwork(layer_dims, activation_functions)
nn.print_model_summary()


costs, accuracies, val_accuracy = nn.train(X_train, Y_train, X_dev, Y_dev, 
                            learning_rate=0.01, 
                            num_epochs=10,
                            batch_size=32,
                            print_cost=True)

# Make predictions
# predictions = nn.predict(X_dev)
# print(nn.get_accuracy(predictions, Y_dev))

plt.figure(figsize=(12, 4))

# training cost
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.title('Training Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')


# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Train')
plt.plot(val_accuracy, label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()