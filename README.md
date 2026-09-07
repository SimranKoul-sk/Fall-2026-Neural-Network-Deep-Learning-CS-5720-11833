# Fall-2026-Neural-Network-Deep-Learning-CS-5720-11833

## Student Information

- **Name:** Simran Koul
- **Student ID:** 700809605
- **Course:** Neural Network & Deep Learning (CS-5720-11833)
- **Assignment:** Part II — Programming
- **Date:** September 11, 2026

## Overview

This repository contains the Python code for Part II of the Neural Network & Deep Learning assignment. The program uses TensorFlow and Matplotlib to demonstrate tensor operations, loss functions, MNIST model training with two optimizers, and TensorBoard logging.

The assignment is divided into four parts:

1. Tensor reshaping and broadcasting
2. Loss function comparison
3. MNIST training with Adam and SGD
4. MNIST training with TensorBoard logging

## Requirements

Install the required libraries before running the code:

```bash
pip install tensorflow matplotlib tensorboard
```

## How to Run

1. Clone this repository or download the project files.
2. Open a terminal in the project folder.
3. Run the Python file:

```bash
python assignment.py
```

The program prints tensor shapes, ranks, and loss values in the terminal. It also opens two Matplotlib charts:

- A bar chart comparing MSE and Categorical Cross-Entropy loss values
- An accuracy chart comparing Adam and SGD training and validation accuracy

After the TensorBoard model finishes training, log files are created in the `logs/fit/` folder.

## Part 1: Tensor Reshaping and Operations

The first section creates a random TensorFlow tensor with shape `(4, 6)`. The code prints the original tensor along with its rank and shape.

The tensor is then reshaped into `(2, 3, 4)`. Reshaping changes how the same 24 values are arranged without changing the actual values because:

```text
4 × 6 = 2 × 3 × 4 = 24
```

Next, the reshaped tensor is transposed from `(2, 3, 4)` to `(3, 2, 4)`. The code swaps the first two axes using:

```python
tf.transpose(reshaped_tensor, perm=(1, 0, 2))
```

### Broadcasting

The program creates a smaller tensor with shape `(1, 4)`:

```python
[[10, 20, 30, 40]]
```

It then adds this smaller tensor to the transposed tensor with shape `(3, 2, 4)`. TensorFlow automatically uses broadcasting so the two tensors can be added.

TensorFlow compares tensor dimensions starting from the right side:

- The last dimension is size `4` for both tensors, so it already matches.
- The smaller tensor has a dimension of size `1`, which TensorFlow repeats to match size `2`.
- The missing leading dimension is treated as size `1` and repeated to match size `3`.

Because of this, TensorFlow can treat the smaller `(1, 4)` tensor as if it had shape `(3, 2, 4)` during addition. The original smaller tensor is not manually copied in the code.

## Part 2: Loss Functions and Hyperparameter Tuning

The second section compares two common loss functions:

- **Mean Squared Error (MSE):** Measures the average squared difference between the true values and predicted values.
- **Categorical Cross-Entropy (CCE):** Measures how well predicted class probabilities match the correct one-hot encoded class labels.

The program defines one-hot encoded true labels for three classes and creates two sets of predictions:

- `y_pred_good` contains predictions that are closer to the correct classes.
- `y_pred_changed` contains less confident and less accurate predictions.

The code calculates and prints MSE and CCE for both prediction sets. When the predictions become less accurate, the loss values increase.

A Matplotlib bar chart is displayed to compare MSE and Cross-Entropy for the better predictions and the changed predictions.

## Part 3: MNIST Training With Adam and SGD

The third section loads the MNIST handwritten-digit dataset from TensorFlow. MNIST images are 28 × 28 pixels, and each image belongs to one of 10 digit classes from 0 through 9.

The pixel values are normalized from the range 0–255 to the range 0–1. This helps the neural network train more smoothly.

To keep the program faster to run, the code trains on the first 10,000 training images:

```python
x_train_small = x_train[:10000]
y_train_small = y_train[:10000]
```

Both models use the same neural-network structure so the optimizer comparison is fair:

- A `Flatten` layer converts each 28 × 28 image into one input vector.
- A dense hidden layer with 128 ReLU neurons learns features from the image pixels.
- A dropout layer with a rate of 0.2 helps reduce overfitting by randomly turning off some neurons during training.
- A final dense layer with 10 softmax neurons produces a probability for each digit class.

### Adam Model

The first model uses the Adam optimizer with a learning rate of 0.001:

```python
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
```

### SGD Model

The second model uses Stochastic Gradient Descent (SGD) with a learning rate of 0.01 and momentum of 0.9:

```python
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

Both models are trained for 5 epochs. The program then displays a graph comparing:

- Adam training accuracy
- Adam validation accuracy
- SGD training accuracy
- SGD validation accuracy

The exact results can vary slightly between runs, but the plot makes it easy to compare how each optimizer learns over the five epochs.

## Part 4: TensorBoard Logging

The last section creates another MNIST model and trains it with the Adam optimizer for 5 epochs. A TensorBoard callback is added so TensorFlow saves training information during each epoch.

The logs are saved in a unique timestamped folder inside:

```text
logs/fit/
```

The callback used in the program is:

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)
```

This saves training and validation information, including loss and accuracy, that can be viewed in TensorBoard.

### Launch TensorBoard

After running the Python program, open a terminal in the project folder and run:

```bash
tensorboard --logdir logs/fit
```

TensorBoard will provide a local web address in the terminal. Open that address in a browser to view the available dashboards.

In TensorBoard, the Scalars tab can be used to view:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

The Histograms tab can also show how model weights change during training because the code uses `histogram_freq=1`.

## Files

- `assignment.py` — Main Python source code for all four tasks
- `README.md` — Project documentation, assignment explanation, and student information
- `logs/fit/` — TensorBoard event logs created after the TensorBoard model runs
