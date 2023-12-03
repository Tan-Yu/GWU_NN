# Project Title

## Furniture Detection with TensorFlow

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

Feel free to explore the code and adapt it to your specific needs. Experiment with different model architectures, hyperparameters, and training strategies to achieve optimal results for your furniture detection task.

### Contributing

If you'd like to contribute to this project, please fork the repository and submit pull requests. We welcome any improvements, bug fixes, or additional features.

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### Acknowledgments

Special thanks to the open-source community for providing valuable resources and tools used in this project.
