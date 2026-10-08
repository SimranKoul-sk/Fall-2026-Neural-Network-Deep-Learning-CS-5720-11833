"""Question 1: Implement Convolution from Scratch"""

import numpy as np

#Configuration
#Stride and padding required by the assignment. Both are parameters of
#convolve2d so part (f) can be demonstrated by running stride 2 as well.
STRIDE = 1
PADDING = 0



#Part (a): Store the input and filter using NumPy arrays

#Return the 5x5 input matrix given in the assignment.
def build_input_matrix():
    return np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype=int,
    )


#Return the 3x3 filter given in the assignment.
def build_filter():
    return np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=int,
    )



#Part (b): Slide the filter over the input matrix
#Part (c): Compute the dot product at every valid location

#Return the input surrounded by a border of zeros, or unchanged when padding is 0.
def apply_padding(matrix, padding):
    if padding == 0:
        return matrix
    return np.pad(matrix, pad_width=padding, mode="constant", constant_values=0)


#Slide the filter over the input and return the feature map, without any built-in convolution.
def convolve2d(input_matrix, kernel, stride=1, padding=0):
    padded = apply_padding(input_matrix, padding)
    input_size = padded.shape[0]
    kernel_size = kernel.shape[0]

    #Standard output size formula: (N - F + 2P) / S + 1. Integer division
    #discards positions where the filter would hang off the edge.
    output_size = (input_size - kernel_size) // stride + 1
    feature_map = np.zeros((output_size, output_size), dtype=int)

    #Two loops walk the top-left corner of the window across the input.
    #Multiplying by the stride is what makes the window skip positions.
    for out_row in range(output_size):
        for out_col in range(output_size):
            row_start = out_row * stride
            col_start = out_col * stride

            #The patch of the input currently sitting under the filter.
            region = padded[
                row_start : row_start + kernel_size,
                col_start : col_start + kernel_size,
            ]

            #The dot product: multiply element by element, then sum.
            #np.sum(region * kernel) is the same as (region * kernel).sum().
            feature_map[out_row, out_col] = np.sum(region * kernel)

    return feature_map



#Part (d): Print the complete output feature map
#Part (e): Print the output shape

#Print a labelled matrix with aligned columns.
def print_matrix(title, matrix):
    print(f"\n{title}")
    for row in matrix:
        print("  " + "  ".join(f"{value:3d}" for value in row))


#Print a feature map together with its shape.
def print_feature_map(title, feature_map):
    print_matrix(title, feature_map)
    print(f"Output shape: {feature_map.shape[0]} x {feature_map.shape[1]}")



#Part (f): Explain how changing the stride from 1 to 2 affects the output

#Print the written explanation of the effect of stride, using both computed results.
def explain_stride(stride1_map, stride2_map):
    print("\n" + "=" * 60)
    print("Part (f): Effect of changing the stride from 1 to 2")
    print("=" * 60)
    print(
        "The stride is how far the filter moves between positions. At stride 1 "
        "the window shifts one column at a time and visits every valid location, "
        "so a 5x5 input with a 3x3 filter gives "
        f"{stride1_map.shape[0]}x{stride1_map.shape[1]} outputs. At stride 2 the "
        "window skips every other position, so it only starts at columns 0 and 2 "
        f"and rows 0 and 2, giving {stride2_map.shape[0]}x{stride2_map.shape[1]} "
        "outputs.\n\n"
        "The same formula predicts both: (N - F + 2P) / S + 1, rounded down. "
        "With N=5, F=3, P=0 that is (5 - 3) / 1 + 1 = 3 for stride 1 and "
        "(5 - 3) / 2 + 1 = 2 for stride 2.\n\n"
        "A larger stride downsamples the feature map, which cuts the amount of "
        "computation and memory in the next layer and widens the receptive field "
        "of later layers. The cost is lost spatial detail, because positions the "
        "filter skipped are never measured. Note that the stride 2 output is not "
        "a summary of the stride 1 output, it is a subset of it: every value in "
        "the stride 2 map also appears in the stride 1 map, at the positions that "
        "both strides happened to visit."
    )


#Build the input and filter, convolve at stride 1 and 2, then explain the difference.
def main():
    print("=" * 60)
    print("Question 1: 2-D Convolution Implemented from Scratch")
    print("=" * 60)

    input_matrix = build_input_matrix()
    kernel = build_filter()

    print_matrix("Input matrix (5x5):", input_matrix)
    print_matrix("Filter (3x3):", kernel)
    print(f"\nStride = {STRIDE}, Padding = {PADDING}")

    #The required configuration: stride 1, padding 0.
    feature_map = convolve2d(input_matrix, kernel, stride=STRIDE, padding=PADDING)
    print("\n" + "=" * 60)
    print("Output Feature Map")
    print("=" * 60)
    print_feature_map(f"Stride = {STRIDE}, Padding = {PADDING}:", feature_map)

    #Computed only to support the written answer to part (f).
    stride2_map = convolve2d(input_matrix, kernel, stride=2, padding=PADDING)
    print_feature_map("Stride = 2, Padding = 0 (for comparison):", stride2_map)

    explain_stride(feature_map, stride2_map)


if __name__ == "__main__":
    main()
