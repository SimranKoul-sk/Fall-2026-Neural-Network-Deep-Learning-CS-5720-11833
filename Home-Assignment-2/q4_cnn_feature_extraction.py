"""Question 4: CNN Feature Extraction with Filters and Pooling"""

import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#Task 1: Implement Edge Detection Using Convolution

#Build a 200x200 grayscale image with shapes that produce clear edges.
def create_sample_image():
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (110, 110), color=200, thickness=-1)
    cv2.circle(image, (145, 145), 40, color=255, thickness=-1)
    cv2.line(image, (20, 180), (90, 130), color=150, thickness=3)
    return image


#Load an image as grayscale, or fall back to the generated sample.
def load_grayscale_image(image_path=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image, f"loaded from {image_path}"
        print(f"Could not read '{image_path}'. Using the generated sample image.")
    return create_sample_image(), "generated sample image"


#Apply the assignment's Sobel kernels in the x and y directions.
def apply_sobel_filters(image):
    sobel_x_kernel = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float32,
    )
    sobel_y_kernel = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float32,
    )

    print("Sobel X kernel:")
    print(sobel_x_kernel)
    print("\nSobel Y kernel:")
    print(sobel_y_kernel)

    #CV_64F keeps the result signed: dark-to-light edges are positive and
    #light-to-dark are negative, and uint8 would clip the negatives to zero.
    sobel_x = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=sobel_x_kernel)
    sobel_y = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=sobel_y_kernel)

    return sobel_x, sobel_y


#Show the original beside both Sobel results and save the figure.
def display_edge_results(original, sobel_x, sobel_y):
    images = [original, np.absolute(sobel_x), np.absolute(sobel_y)]
    titles = [
        "Original Image",
        "Edge Detection: Sobel-X",
        "Edge Detection: Sobel-Y",
    ]

    figure, axes = plt.subplots(1, 3, figsize=(13, 5))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle("Task 1: Edge Detection using Sobel Filters")
    figure.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, "q4_sobel_edge_detection.png")
    figure.savefig(output_path, dpi=120)
    print(f"\nSaved figure to: {output_path}")

    plt.show()


#Run the full Sobel edge-detection task.
def run_task1(image_path=None):
    print("=" * 60)
    print("Task 1: Edge Detection using Sobel Filter")
    print("=" * 60)

    image, source = load_grayscale_image(image_path)
    print(f"Grayscale image ({source}), shape: {image.shape}\n")

    sobel_x, sobel_y = apply_sobel_filters(image)

    print(f"\nSobel-X response range: {sobel_x.min():.1f} to {sobel_x.max():.1f}")
    print(f"Sobel-Y response range: {sobel_y.min():.1f} to {sobel_y.max():.1f}")

    display_edge_results(image, sobel_x, sobel_y)


#Task 2: Implement Max Pooling and Average Pooling

#Apply 2x2 max pooling and 2x2 average pooling to a random 4x4 matrix.
def run_task2():
    print("\n" + "=" * 60)
    print("Task 2: Pooling Operations on a Random 4x4 Matrix")
    print("=" * 60)

    #Fixed seed so results can be reproduced between runs.
    np.random.seed(42)
    matrix = np.random.randint(0, 10, size=(4, 4)).astype(np.float32)

    input_tensor = tf.reshape(tf.constant(matrix), [1, 4, 4, 1])

    #The default stride equals the pool size, so the 2x2 windows do not overlap.
    max_pool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    average_pool_layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))

    max_pooled = max_pool_layer(input_tensor).numpy()[0, :, :, 0]
    average_pooled = average_pool_layer(input_tensor).numpy()[0, :, :, 0]

    #Print one labelled matrix with its shape.
    def show(title, array):
        print(f"\n{title} (shape {array.shape[0]}x{array.shape[1]}):")
        for row in array:
            print("  " + "  ".join(f"{value:6.2f}" for value in row))

    show("Original 4x4 matrix", matrix)
    show("Max-pooled matrix (2x2 pooling)", max_pooled)
    show("Average-pooled matrix (2x2 pooling)", average_pooled)

    print(
        "\nMax pooling keeps the single strongest value in each 2x2 window, which preserves the most prominent feature and discards the rest. Average pooling keeps the mean of each window, which smooths the response and retains more background information but dilutes sharp features."
    )


#Run both tasks in order.
def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_task1(image_path)
    run_task2()


if __name__ == "__main__":
    main()
