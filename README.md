Project in the works, check back soon for updates.

To run Tensorflow on docker with GPU support, use the following command:

```bash
docker run -it --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu python
```

To run Tensorflow on docker with CPU support, use the following command:

```bash
docker run -it tensorflow/tensorflow bash
```

### Model Architecture Overview

```bash
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ rescaling (Rescaling)           │ (None, 256, 256, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 253, 253, 16)   │           784 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 126, 126, 16)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 123, 123, 32)   │         8,224 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 61, 61, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 61, 61, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 58, 58, 64)     │        32,832 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 29, 29, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 29, 29, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 26, 26, 128)    │       131,200 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 13, 13, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 21632)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │     2,769,024 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 8)              │         1,032 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2,943,096 (11.23 MB)
 Trainable params: 2,943,096 (11.23 MB)
 Non-trainable params: 0 (0.00 B)
```

This document provides an overview of the architecture of a convolutional
neural network (CNN) model designed for image classification.
Each layer of the model is explained in terms of its purpose,
output shape, and the number of parameters (`Param #`) it utilizes.

#### Layer-by-Layer Explanation

#### 1. Rescaling Layer (`rescaling`)

- **Purpose**: Normalize pixel values of input images to the range [0, 1].
- **Output Shape**: `(None, 256, 256, 3)`
- **Param #**: `0`
    - No trainable parameters; it's a preprocessing step.

#### 2. Convolutional Layer (`conv2d`)

- **Purpose**: Apply convolution operation with 16 filters.
- **Output Shape**: `(None, 253, 253, 16)`
- **Param #**: `784`
    - **Explanation**:
        - **Convolution**: This layer convolves (slides) 16 filters (small
          matrices) across the input image to produce 16 feature maps.
        - **Kernel Size**: The size of each filter matrix is `4x4`
          pixels (`kernel_size=4`), which determines the local input region to
          which each filter is applied.
        - **Activation Function (ReLU)**: Applies the ReLU (Rectified Linear
          Unit) activation function element-wise to introduce non-linearity.

#### 3. MaxPooling2D Layer (`max_pooling2d`)

- **Purpose**: Downsample representation by extracting maximum values.
- **Output Shape**: `(None, 126, 126, 16)`
- **Param #**: `0`
    - **Explanation**:
        - **Max Pooling**: Reduces the dimensionality of each feature map,
          retaining the most important information.
        - **Pool Size**: Defaults to a `2x2` window (`pool_size=(2, 2)`),
          halving the spatial dimensions (width and height).

#### 4. Convolutional Layer (`conv2d_1`)

- **Purpose**: Apply convolution operation with 32 filters.
- **Output Shape**: `(None, 123, 123, 32)`
- **Param #**: `8,224`
    - **Explanation**:
        - Builds upon the previous convolutional layer, extracting 32 different
          features.

#### 5. MaxPooling2D Layer (`max_pooling2d_1`)

- **Purpose**: Downsample representation.
- **Output Shape**: `(None, 61, 61, 32)`
- **Param #**: `0`

#### 6. Dropout Layer (`dropout`)

- **Purpose**: Regularization to prevent overfitting by randomly setting a
  fraction of input units to zero.
- **Output Shape**: `(None, 61, 61, 32)`
- **Param #**: `0`

#### 7. Convolutional Layer (`conv2d_2`)

- **Purpose**: Apply convolution operation with 64 filters.
- **Output Shape**: `(None, 58, 58, 64)`
- **Param #**: `32,832`

#### 8. MaxPooling2D Layer (`max_pooling2d_2`)

- **Purpose**: Further downsample representation.
- **Output Shape**: `(None, 29, 29, 64)`
- **Param #**: `0`

#### 9. Dropout Layer (`dropout_1`)

- **Purpose**: Regularization to prevent overfitting.
- **Output Shape**: `(None, 29, 29, 64)`
- **Param #**: `0`

#### 10. Convolutional Layer (`conv2d_3`)

- **Purpose**: Apply convolution operation with 128 filters.
- **Output Shape**: `(None, 26, 26, 128)`
- **Param #**: `131,200`

#### 11. MaxPooling2D Layer (`max_pooling2d_3`)

- **Purpose**: Further downsample representation.
- **Output Shape**: `(None, 13, 13, 128)`
- **Param #**: `0`

#### 12. Flatten Layer (`flatten`)

- **Purpose**: Convert 2D matrix into a vector.
- **Output Shape**: `(None, 21632)`
- **Param #**: `0`

#### 13. Dense Layer (`dense`)

- **Purpose**: Fully connected layer with 128 neurons.
- **Output Shape**: `(None, 128)`
- **Param #**: `2,769,024`

#### 14. Dense Layer (`dense_1`)

- **Purpose**: Output layer with 8 neurons (equal to the number of classes).
- **Output Shape**: `(None, 8)`
- **Param #**: `1,032`

---

## Summary

This CNN model uses a series of convolutional and pooling layers followed by
fully connected layers for image classification. Here's a breakdown of key
terms:

- **Convolutional Layer**: Applies filters to input images to extract features
  like edges and textures.
- **Max Pooling**: Reduces the spatial dimensions of each feature map, focusing
  on the most important features.
- **Kernel Size**: Specifies the size of the filter matrix used in
  convolutional layers.
- **Dropout**: Randomly drops a fraction of connections between layers during
  training to prevent overfitting.
- **Activation Function (ReLU)**: Introduces non-linearity to the model,
  allowing it to learn complex patterns in the data.

Understanding these components helps in designing effective neural networks for
tasks like image classification, improving both model accuracy and efficiency.
