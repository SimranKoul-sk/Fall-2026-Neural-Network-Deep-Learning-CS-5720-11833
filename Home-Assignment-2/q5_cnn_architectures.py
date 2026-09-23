"""Question 5: Implementing and Comparing CNN Architectures"""

import tensorflow as tf
from tensorflow.keras import layers, models

#Task 1: Implement AlexNet Architecture

def build_alexnet(input_shape=(227, 227, 3), num_classes=10):
    model = models.Sequential(name="Simplified_AlexNet")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=4, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    #Conv layers 2-5 use padding='same' as in the original AlexNet, which keeps
    #the classic 6x6x256 feature map before the flatten.
    model.add(layers.Conv2D(256, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(layers.Conv2D(384, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(384, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation="softmax"))

    return model

#Task 2: Implement a Residual Block and ResNet

def residual_block(input_tensor, filters):
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation="relu")(
        input_tensor
    )
    #The second convolution is left linear so the skip connection is added
    #before the activation, as in the ResNet paper.
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = layers.Add()([x, input_tensor])
    x = layers.Activation("relu")(x)
    return x


def build_resnet(input_shape=(64, 64, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        64, kernel_size=(7, 7), strides=2, padding="same", activation="relu"
    )(inputs)

    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="Simple_ResNet")


def main():
    print("=" * 70)
    print("Task 1: Simplified AlexNet")
    print("=" * 70)
    alexnet = build_alexnet()
    alexnet.summary()

    print("\n" + "=" * 70)
    print("Task 2: ResNet-like model with Residual Blocks")
    print("=" * 70)
    resnet = build_resnet()
    resnet.summary()

    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(
        "AlexNet is a plain stack: every layer feeds only the next one. Almost all of its parameters sit in the two 4096-neuron dense layers rather than in the convolutions, which is why the dropout is placed there.\n\n"
        "The ResNet model adds skip connections, so each block only has to learn the residual (the change) rather than a full transformation. If a block is not useful it can learn to output roughly zero and pass its input through unchanged. The addition also gives gradients a direct path backwards, which is what allows ResNets to be made much deeper than AlexNet without the gradient vanishing on the way back."
    )


if __name__ == "__main__":
    main()
