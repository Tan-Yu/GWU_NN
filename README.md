# Furniture Detection with TensorFlow

### Overview

This Git repository contains code for a furniture detection project using TensorFlow. The project involves importing necessary libraries, preparing the dataset, and experimenting with different convolutional neural network (CNN) models for furniture detection. The goal is to identify and locate furniture items within images.

### Getting Started

To get started with the project, follow the instructions below:

#### Import Libraries

Make sure you have the required libraries installed. You can install them using the following:

```bash
pip install opencv-python numpy pandas tensorflow matplotlib
```

#### Data Preparation

1. Set the dataset path by updating the `data_path` variable in the script to point to your dataset directory.

```python
data_path = "/path/to/your/dataset"
```

2. Load annotations for the training and validation sets using the following:

```python
train_annotations = pd.read_csv(os.path.join(data_path, "train", "_annotations.csv"))
valid_annotations = pd.read_csv(os.path.join(data_path, "valid", "_annotations.csv"))
```

3. Rescale bounding box coordinates to ensure consistency between annotations and image size:

```python
train_annotations = train_annotations.apply(rescale_bbox, axis=1)
valid_annotations = valid_annotations.apply(rescale_bbox, axis=1)
```

4. Create an ImageDataGenerator for data augmentation and custom data generators for the training and validation sets:

```python
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(...)
valid_generator = datagen.flow_from_dataframe(...)
```

### Experimenting with Models

The repository includes experiments with various CNN models for furniture detection. The script explores different architectures, including convolutional layers, pooling, flattening, dense layers, dropout, batch normalization, and more. Additionally, it employs the Adam optimizer, early stopping, and learning rate reduction techniques for model training.
# Experimented Models

## Model #1

**Input Layer:**
- Input shape: (300, 300, 3) representing a 300x300 pixel image with 3 color channels (RGB).

**Convolutional Layers:**
- Conv2D(64, (3, 3), activation='relu'): 64 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling with a 2x2 pool size to down-sample the spatial dimensions.
- Dropout(0.25): Regularization layer to prevent overfitting.
- Conv2D(128, (3, 3), activation='relu'): 128 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.
- Conv2D(256, (3, 3), activation='relu'): 256 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.

**Flatten Layer:**
- Flattens the output from the previous layers into a 1D array to be fed into dense layers.

**Fully Connected (Dense) Layers:**
- Dense(512, activation='relu'): 512 neurons with ReLU activation.
- Dropout(0.5): Another dropout layer for regularization to further prevent overfitting.

**Output Layer:**
- Dense(4, activation='linear'): Output layer with 4 neurons corresponding to bounding box coordinates (xmin, ymin, xmax, ymax) for regression. Linear activation is used since this is a regression task.

**Key Points:**
- Convolutional layers extract hierarchical features from the input image.
- Max pooling layers down-sample the spatial dimensions.
- Dropout layers help prevent overfitting.
- The final dense layers map the extracted features to bounding box coordinates.
- Smooth L1 loss is typically used for regression tasks like bounding box prediction.
- This architecture aims to balance complexity for feature extraction and regularization to create an effective model for object detection tasks.

## Model #2

**Input Layer:**
- Input shape: (300, 300, 3) representing a 300x300 pixel image with 3 color channels (RGB).

**Feature Extraction Backbone:**
- Conv2D(32, (3, 3), activation='relu'): 32 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling with a 2x2 pool size to down-sample the spatial dimensions.
- Dropout(0.25): Regularization layer to prevent overfitting.
- Conv2D(64, (3, 3), activation='relu'): 64 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.
- Dropout(0.25): Regularization layer.
- Conv2D(128, (3, 3), activation='relu'): 128 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.
- Dropout(0.25): Regularization layer.
- Conv2D(256, (3, 3), activation='relu'): 256 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.
- Dropout(0.25): Regularization layer.
- Conv2D(512, (3, 3), activation='relu'): 512 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.
- Dropout(0.25): Regularization layer.
- Conv2D(512, (3, 3), activation='relu'): Another set of 512 filters with ReLU activation.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.
- Dropout(0.25): Regularization layer.

**Global Average Pooling Layer:**
- GlobalAveragePooling2D(): Aggregates spatial information by computing the average of each feature map.

**Fully Connected (Dense) Layers:**
- Dense(512, activation='relu'): 512 neurons with ReLU activation.
- Dropout(0.5): Regularization layer to prevent overfitting.

**Output Layer:**
- Dense(4, activation='linear'): Output layer with 4 neurons corresponding to bounding box coordinates (xmin, ymin, xmax, ymax) for regression. Linear activation is used since this is a regression task.

**Key Points:**
- The architecture uses multiple convolutional layers for hierarchical feature extraction.
- Dropout layers are employed for regularization.
- Global Average Pooling is used to reduce spatial dimensions before fully connected layers.
- Dense layers map the features to bounding box coordinates.
- Linear activation is used in the output layer for regression.
- This architecture is designed for object detection with bounding box regression and emphasizes feature extraction through a deep convolutional backbone.

## Model #3: Batch Normalization and Regularization

**Input Layer:**
- Input shape: (300, 300, 3) representing a 300x300 pixel image with 3 color channels (RGB).

**Convolutional Blocks:**
- Conv2D(32, (3, 3), activation='relu'): 32 filters of size 3x3, ReLU activation.
- BatchNormalization(): Normalizes the activations of the previous layer for each batch.
- MaxPooling2D((2, 2)): Max pooling with a 2x2 pool size to down-sample spatial dimensions.
- Dropout(0.25): Regularization layer to prevent overfitting.
- Conv2D(64, (3, 3), activation='relu'): 64 filters of size 3x3, ReLU activation.
- BatchNormalization(): Normalization layer.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.
- Dropout(0.25): Regularization layer.
- Conv2D(128, (3, 3), activation='relu'): 128 filters of size 3x3, ReLU activation.
- BatchNormalization(): Normalization layer.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.
- Dropout(0.25): Regularization layer.

**Flatten Layer:**
- Flatten(): Flattens the input to prepare for fully connected layers.

**Fully Connected (Dense) Layers:**
- Dense(256, activation='relu', kernel_regularizer='l2'): 256 neurons with ReLU activation and L2 regularization.
- BatchNormalization(): Normalization layer.
- Dropout(0.5): Regularization layer to prevent overfitting.

**Output Layer:**
- Dense(4, activation='linear'): Output layer with 4 neurons corresponding to bounding box coordinates (xmin, ymin, xmax, ymax) for regression. Linear activation is used since this is a regression task.

**Key Points:**
- Convolutional blocks include Batch Normalization after each convolutional and pooling layer.
- L2 regularization is applied to the first fully connected layer.
- Dropout layers are used for regularization throughout the network.
- The architecture aims to improve training stability and convergence through normalization techniques (Batch Normalization) and regularization

.
- This architecture incorporates batch normalization to stabilize and accelerate the training process and dropout layers for regularization. L2 regularization is applied to the first fully connected layer to control overfitting.

## Model #4

**Input Layer:**
- Input shape: (300, 300, 3) representing a 300x300 pixel image with 3 color channels (RGB).

**Convolutional Blocks:**
- Conv2D(32, (3, 3), activation='relu'): 32 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling with a 2x2 pool size to down-sample spatial dimensions.
- Conv2D(64, (3, 3), activation='relu'): 64 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.

**Flatten Layer:**
- Flatten(): Flattens the input to prepare for fully connected layers.

**Fully Connected (Dense) Layers:**
- Dense(128, activation='relu'): 128 neurons with ReLU activation.
- Dense(4, activation='linear'): Output layer with 4 neurons corresponding to bounding box coordinates (xmin, ymin, xmax, ymax) for regression. Linear activation is used since this is a regression task.

**Key Points:**
- Convolutional blocks followed by max-pooling layers are used for feature extraction and down-sampling.
- The Flatten layer prepares the data for fully connected layers.
- The first fully connected layer has 128 neurons with ReLU activation.
- The output layer has 4 neurons with linear activation for bounding box regression.

## Model #5

**Input Layer:**
- Input shape: (300, 300, 3) representing a 300x300 pixel image with 3 color channels (RGB).

**Convolutional Blocks:**
- Conv2D(32, (3, 3), activation='relu'): 32 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling with a 2x2 pool size to down-sample spatial dimensions.
- Conv2D(64, (3, 3), activation='relu'): 64 filters of size 3x3, ReLU activation.
- MaxPooling2D((2, 2)): Max pooling for down-sampling.

**Flatten Layer:**
- Flatten(): Flattens the input to prepare for fully connected layers.

**Fully Connected (Dense) Layers:**
- Dense(256, activation='relu'): 256 neurons with ReLU activation.
- Dense(128, activation='relu'): 128 neurons with ReLU activation.
- Dense(64, activation='relu'): 64 neurons with ReLU activation.
- Dense(4, activation='linear'): Output layer with 4 neurons corresponding to bounding box coordinates (xmin, ymin, xmax, ymax) for regression. Linear activation is used since this is a regression task.

**Key Points:**
- Convolutional blocks followed by max-pooling layers are used for feature extraction and down-sampling.
- The Flatten layer prepares the data for fully connected layers.
- Additional dense layers (256, 128, 64 neurons) are added to enhance the capacity of the model.
- The output layer remains with 4 neurons for bounding box regression with linear activation.

**Note:**
- The additional dense layers may capture more complex patterns but could also lead to overfitting, so regularization techniques or adjustments may be needed based on specific dataset characteristics and training performance.


### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### Acknowledgments

Special thanks to the open-source community for providing valuable resources and tools used in this project.
