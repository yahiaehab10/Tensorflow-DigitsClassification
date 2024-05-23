# Machine Learning vs Deep Learning

## Table of Contents
- [Introduction](#introduction)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Feature Extraction with SIFT and Bag-of-Words](#feature-extraction-with-sift-and-bag-of-words)
- [Deep Learning Classification](#deep-learning-classification)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction
This project compares traditional machine learning techniques with deep learning methods for image classification tasks using the MNIST dataset. It includes:
- Data collection and preprocessing using PyTorch.
- Feature extraction with SIFT and Bag-of-Words.
- Deep learning classification using Convolutional Neural Networks (CNNs).

## Data Collection and Preprocessing
The dataset used is the MNIST dataset, which consists of handwritten digit images. The data is downloaded and preprocessed using PyTorch.

### Imports
```python
import torch as th
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import tqdm
```

### Downloading & Labeling Data
```python
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None,
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None,
)
```

### Showing Sample of Data
```python
fig, axes = plt.subplots(1, 5, figsize=(15, 15))
for i in range(5):
    idx = np.random.randint(0, len(train_data))
    image, label = train_data[idx]
    axes[i].imshow(image.squeeze(), cmap="gray")
    axes[i].set_title(train_data.classes[label])
    axes[i].axis(False)
```

## Feature Extraction with SIFT and Bag-of-Words
This section covers feature extraction using the SIFT algorithm and the creation of Bag-of-Words histograms.

### Methodology:
1. Convert PyTorch tensor images to 8-bit numpy arrays.
2. Convert dataset images to numpy arrays.
3. Use SIFT feature extractor.
4. Extract SIFT features from images.
5. Create histograms of visual words.

### Code Example:
```python
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import torch

def tensor_to_np(image_tensor):
    image = image_tensor.numpy()
    image = np.moveaxis(image, 0, -1)
    image = (image * 255).astype(np.uint8)
    return image

sift = cv2.SIFT_create()

def extract_sift_features(images):
    descriptors = []
    for image in images:
        kp, desc = sift.detectAndCompute(image, None)
        if desc is not None:
            descriptors.append(desc)
    return np.vstack(descriptors)

# Further processing ...
```

## Deep Learning Classification
A CNN model is built and trained on the MNIST dataset.

### Model Creation and Training
```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

X_train_cnn = np.array([image.numpy().reshape(28, 28, 1) for image, label in train_data])
y_train_cnn = np.array([label for image, label in train_data])
X_test_cnn = np.array([image.numpy().reshape(28, 28, 1) for image, label in test_data])
y_test_cnn = np.array([label for image, label in test_data])

history = model.fit(X_train_cnn, y_train_cnn, epochs=5, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cnn, verbose=2)
print(f"Test accuracy: {test_acc}")
```

## Evaluation and Visualization
The model's performance is evaluated using a confusion matrix and classification report.

### Code Example:
```python
y_pred_cnn = np.argmax(model.predict(X_test_cnn), axis=-1)
print("CNN Classification Report:\n", classification_report(y_test_cnn, y_pred_cnn))

conf_matrix = confusion_matrix(y_test_cnn, y_pred_cnn)

plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=train_data.classes,
    yticklabels=train_data.classes,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for CNN")
plt.show()
```

## Installation
To install the required dependencies, use the following commands:
```sh
pip install torch torchvision matplotlib tqdm scikit-learn tensorflow
```

## Usage
1. Clone the repository.
2. Install the required dependencies as mentioned above.
3. Run the `notebook.ipynb` notebook to execute the code.

## Project Structure
- `notebook.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, feature extraction, model training, and evaluation.

## License
This project was given and managed by German International University
