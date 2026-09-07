import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

#Make random values repeatable at every run
tf.random.set_seed(42)

#1. Tensor Reshaping & Operations

#Create the original 2D tensor
#4 rows x 6 columns = 24 elements.
original_tensor = tf.random.uniform(shape=(4, 6), minval=0, maxval=10, dtype=tf.int32)

#tf.rank = the number of dimensions
#tf.shape = dimension sizes
print("Original tensor:")
print(original_tensor.numpy())
print("Rank:", tf.rank(original_tensor).numpy())
print("Shape:", tf.shape(original_tensor).numpy())

#Reshaping w/o changing data
#4 * 6 = 2 * 3 * 4 = 24 elements
reshaped_tensor = tf.reshape(original_tensor, shape=(2, 3, 4))

print("\nReshaped tensor (2, 3, 4):")
print(reshaped_tensor.numpy())
print("Rank:", tf.rank(reshaped_tensor).numpy())
print("Shape:", tf.shape(reshaped_tensor).numpy())

#Reorder the first 2 axis
#The last axis (size 4) remains the same
transposed_tensor = tf.transpose(reshaped_tensor, perm=(1, 0, 2))

print("\nTransposed tensor (3, 2, 4):")
print(transposed_tensor.numpy())
print("Rank:", tf.rank(transposed_tensor).numpy())
print("Shape:", tf.shape(transposed_tensor).numpy())

#(1, 4) tensor gives 4 values for final dimension
small_tensor = tf.constant([[10, 20, 30, 40]], dtype=tf.int32)

#TensorFlow gives (1, 4) to (3, 2, 4) during addition
#The missing dimension is treated as 1 and repeated to 3
#The dimension of size 1 is repeated to size 2
#The final dimension already has matching size 4
result_tensor = transposed_tensor + small_tensor

print("\nSmaller tensor (1, 4):")
print(small_tensor.numpy())
print("Shape:", tf.shape(small_tensor).numpy())
print("\nResult after broadcasting and addition (3, 2, 4):")
print(result_tensor.numpy())
print("Rank:", tf.rank(result_tensor).numpy())
print("Shape:", tf.shape(result_tensor).numpy())


#2. Loss Functions & Hyperparameter Tuning

print("Loss Functions")

#True labels are one-hot encoded for 3 different classes
#Each row represents the correct class for one example
y_true = tf.constant([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

#First prediction set is closer to the actual answers
y_pred_good = tf.constant([
    [0.80, 0.10, 0.10],
    [0.10, 0.75, 0.15],
    [0.05, 0.10, 0.85]
])

#Second prediction set is less confident and less accurate
y_pred_changed = tf.constant([
    [0.45, 0.30, 0.25],
    [0.30, 0.40, 0.30],
    [0.25, 0.35, 0.40]
])

#Create loss function objects
#MSE checks the average squared difference between true and predicted values
#CCE checks how well the predicted probability distribution matches the true class
mse_loss = tf.keras.losses.MeanSquaredError()
cce_loss = tf.keras.losses.CategoricalCrossentropy()

#Calculate losses for the better predictions
mse_good = mse_loss(y_true, y_pred_good).numpy()
cce_good = cce_loss(y_true, y_pred_good).numpy()

#Calculate losses after changing the predictions
mse_changed = mse_loss(y_true, y_pred_changed).numpy()
cce_changed = cce_loss(y_true, y_pred_changed).numpy()

print("\nLoss values for better predictions:")
print("MSE:", round(mse_good, 4))
print("Categorical Cross-Entropy:", round(cce_good, 4))

print("\nLoss values for changed predictions:")
print("MSE:", round(mse_changed, 4))
print("Categorical Cross-Entropy:", round(cce_changed, 4))

#Set labels and values for the loss comparison chart
loss_labels = ["MSE", "Cross-Entropy"]
good_losses = [mse_good, cce_good]
changed_losses = [mse_changed, cce_changed]

#Create positions for side-by-side bars
x = range(len(loss_labels))
bar_width = 0.35

#Plot loss values before & after changing predictions
plt.figure(figsize=(9, 5))
plt.bar(
    [value - bar_width / 2 for value in x],
    good_losses,
    width=bar_width,
    label="Better Predictions"
)
plt.bar(
    [value + bar_width / 2 for value in x],
    changed_losses,
    width=bar_width,
    label="Changed Predictions"
)

plt.xticks(list(x), loss_labels)
plt.ylabel("Loss Value")
plt.title("MSE and Categorical Cross-Entropy Loss Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


#3. Train MNIST with Adam and SGD

print("MNIST: Adam vs SGD")

#Load the handwritten digit dataset
#Each image is 28 x 28 pixels and each label is a digit from 0 to 9
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Change pixel values from 0-255 into values between 0 and 1 to train the model smoothly
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train_small = x_train[:10000]
y_train_small = y_train[:10000]

#Create the same model structure for both optimizers to make the comparison fair
def create_mnist_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),

        #Hidden layer learns useful features from the image pixels
        tf.keras.layers.Dense(128, activation="relu"),

        #Dropout randomly turns off some neurons during training
        #This can help lower overfitting
        tf.keras.layers.Dropout(0.2),

        #10 output neurons because MNIST has digits 0 through 9
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    return model

#Create and compile the Adam model
adam_model = create_mnist_model()

#Sparse categorical cross-entropy works with integer labels like 0, 1, 2, etc.
adam_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Train the 1st model using Adam
print("Training model with Adam ")
adam_history = adam_model.fit(
    x_train_small,
    y_train_small,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128,
    verbose=1
)

#Create and compile a new model for SGD so both start with fresh weights
sgd_model = create_mnist_model()

#Momentum helps SGD keep moving in useful directions
sgd_model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Train the second model using SGD
print("Training model with SGD ")
sgd_history = sgd_model.fit(
    x_train_small,
    y_train_small,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128,
    verbose=1
)

#Create epoch numbers for the accuracy chart
epochs = range(1, 6)

#Plot training and validation accuracy for both optimizers
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    adam_history.history["accuracy"],
    marker="o",
    label="Adam Training Accuracy"
)

plt.plot(
    epochs,
    adam_history.history["val_accuracy"],
    marker="o",
    linestyle="--",
    label="Adam Validation Accuracy"
)

plt.plot(
    epochs,
    sgd_history.history["accuracy"],
    marker="s",
    label="SGD Training Accuracy"
)

plt.plot(
    epochs,
    sgd_history.history["val_accuracy"],
    marker="s",
    linestyle="--",
    label="SGD Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST Accuracy Comparison: Adam vs SGD")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("adam_vs_sgd_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()


#4. MNIST Model with TensorBoard

print("TensorBoard Training")

#Create another model for TensorBoard logging
tensorboard_model = create_mnist_model()

#Compile this model with Adam
tensorboard_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Create a unique dir with every code run
log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

#TensorBoard callback saves loss and accuracy after each epoch
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

#Train for the required 5 epochs and save logs for TensorBoard
print("Training TensorBoard model ")
tensorboard_history = tensorboard_model.fit(
    x_train_small,
    y_train_small,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128,
    callbacks=[tensorboard_callback],
    verbose=1
)
#Print the dir location so it is easy to find
print("\nTensorBoard logs saved in:")
print(log_dir)
