# Neural Network Implementation for MNIST Classification

This repository contains various implementations of neural networks for the MNIST digit classification task, ranging from basic to more sophisticated versions.

## Project Structure

- `mnist_2l_basic.py`: Basic implementation of a 2-layer neural network
- `mnist_2l_batch.py`: Enhanced version with batch processing
- `mnist_custom.py`: Modular implementation with customizable architecture
- `mnist_custom_batch.py`: Complete implementation with batch processing and visualization

## Features

- Multiple neural network implementations with increasing complexity
- Customizable network architectures
- Support for different activation functions (ReLU, tanh, sigmoid, softmax)
- Batch processing for improved training efficiency
- Training progress visualization
- Model performance evaluation

## Network Architecture

The default architecture used in the custom implementations:
- Input layer: 784 neurons (28x28 pixel images)
- First hidden layer: 128 neurons with ReLU activation
- Second hidden layer: 64 neurons with ReLU activation
- Output layer: 10 neurons with softmax activation

## Training Features

- Mini-batch gradient descent
- Learning rate customization
- Training/validation split
- Progress monitoring with cost and accuracy metrics
- Live visualization of training progress

## Requirements

- NumPy
- Pandas
- Matplotlib
- MNIST dataset (train.csv)

## Usage

### Basic 2-Layer Network
```python
python mnist_2l_basic.py
```

### Batch Processing Version
```python
python mnist_2l_batch.py
```

### Custom Architecture
```python
python mnist_custom.py
```

### Full Featured Version
```python
python mnist_custom_batch.py
```

### Creating Custom Network Architecture

```python
# Define network architecture
layer_dims = [784, 128, 64, 10]  # Input layer, 2 hidden layers, output layer
activation_functions = ['relu', 'relu', 'softmax']  # Activation functions for each layer

# Create and train network
nn = NeuralNetwork(layer_dims, activation_functions)
costs, accuracies, val_accuracy = nn.train(X_train, Y_train, X_dev, Y_dev, 
                                         learning_rate=0.01, 
                                         num_epochs=10,
                                         batch_size=32)
```

## Model Summary

The neural network implementation provides a detailed model summary including:
- Layer-wise architecture details
- Parameter count for each layer
- Total network parameters
- Activation functions used

## Performance Visualization

The training process includes visualization of:
- Training cost over epochs
- Training accuracy over epochs
- Validation accuracy over epochs

## Note

Make sure to update the path to your MNIST dataset in the code:
```python
data = pd.read_csv('path_to_your_train.csv')
```