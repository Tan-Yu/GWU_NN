# Example usage in your network
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Conv2DLayer

from gwu_nn.activation_layers import Sigmoid
import numpy

import medmnist
from medmnist import DermaMNIST

train = DermaMNIST(split="train", download=True)
x_train = train.imgs
y_train = train.labels.ravel()

test = DermaMNIST(split="test", download=True)
x_test = test.imgs
y_test = test.labels.ravel()
val = DermaMNIST(split="val", download=True)
x_val = val.imgs
y_val = val.labels.ravel()
import numpy as np
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Conv2DLayer
from gwu_nn.activation_layers import RELU, Softmax
from gwu_nn.loss_functions import CrossEntropy



# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0
x_val = x_val / 255.0

# One-hot encode the labels
num_classes = 7  # assuming 7 categories
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]
y_val_onehot = np.eye(num_classes)[y_val]

# Create the neural network
model = GWUNetwork()

# Add Convolutional Layer
model.add(Conv2DLayer(filters=64, kernel_size=(3, 3), activation='relu', input_size=(28, 28, 3)))

# Flatten the output from the convolutional layer
model.add(Dense(output_size=128, activation='relu'))

# Output layer with Softmax activation for classification
model.add(Dense(output_size=num_classes, activation='softmax'))

# Compile the model with CrossEntropy loss and learning rate
model.compile(loss='cross_entropy', lr=0.001)

# Train the model
model.fit(x_train, y_train_onehot, epochs=10, batch_size=32)

# Evaluate the model on the validation set
val_loss = model.evaluate(x_val, y_val_onehot)
print(f'Validation Loss: {val_loss}')

# Make predictions on the test set
predictions = model.predict(x_test)

# You can then use predictions for further analysis or evaluation

