"""Question 3: Convolution Operations with Different Parameters"""

import numpy as np
import tensorflow as tf

#Step 1: Define the 5x5 input matrix

#Return the 5x5 input matrix given in the assignment.
def build_input_matrix():
    return np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        dtype=np.float32,
    )

#Step 2: Define the 3x3 kernel

#Return the 3x3 Laplacian kernel given in the assignment.
def build_kernel():
    return np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

#Step 3: Perform the convolution operations

#Apply one 2D convolution and return the feature map as a 2D array.
def convolve(input_matrix, kernel, stride, padding):
    #tf.nn.conv2d needs 4D tensors: input is (batch, height, width, channels)
    #and the kernel is (k_height, k_width, in_channels, out_channels).
    input_tensor = tf.reshape(
        tf.constant(input_matrix), [1, input_matrix.shape[0], input_matrix.shape[1], 1]
    )
    kernel_tensor = tf.reshape(
        tf.constant(kernel), [kernel.shape[0], kernel.shape[1], 1, 1]
    )
    output = tf.nn.conv2d(
        input_tensor,
        kernel_tensor,
        strides=[1, stride, stride, 1],
        padding=padding,
    )
    return output.numpy()[0, :, :, 0]

#Step 4: Print the output feature maps

#Print a labelled matrix with aligned columns and its shape.
def print_matrix(title, matrix):
    print(f"\n{title}")
    print(f"Output shape: {matrix.shape[0]} x {matrix.shape[1]}")
    for row in matrix:
        print("  " + "  ".join(f"{value:7.1f}" for value in row))


#Run all four stride/padding combinations and print each feature map.
def main():
    input_matrix = build_input_matrix()
    kernel = build_kernel()

    print("=" * 60)
    print("Question 3: Convolution with Different Stride and Padding")
    print("=" * 60)

    print_matrix("Input matrix (5x5):", input_matrix)
    print_matrix("Kernel (3x3):", kernel)

    configurations = [
        (1, "VALID"),
        (1, "SAME"),
        (2, "VALID"),
        (2, "SAME"),
    ]

    print("\n" + "=" * 60)
    print("Output Feature Maps")
    print("=" * 60)

    for stride, padding in configurations:
        feature_map = convolve(input_matrix, kernel, stride, padding)
        print_matrix(f"Stride = {stride}, Padding = '{padding}'", feature_map)

    print("\n" + "=" * 60)
    print("Why the output sizes differ")
    print("=" * 60)
    print(
        "VALID padding adds nothing around the input, so the 3x3 kernel only sits in positions where it fits completely, shrinking a 5x5 input to 3x3 at stride 1. SAME padding adds a border of zeros so the output keeps the input size when the stride is 1, giving 5x5. Increasing the stride to 2 makes the kernel skip every other position, roughly halving each output dimension: 2x2 for VALID and 3x3 for SAME.\n\n"
        "The VALID outputs are all zero because the kernel is a discrete Laplacian and the input is a linear ramp, where every pixel is exactly the average of its four neighbours. Only the zero-padded borders in the SAME results break that linearity and produce non-zero values."
    )


if __name__ == "__main__":
    main()
