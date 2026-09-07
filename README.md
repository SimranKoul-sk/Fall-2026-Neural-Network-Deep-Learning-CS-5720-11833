# Fall-2026-Neural-Network-Deep-Learning-CS-5720-11833

## Student Information
- **Name:** Simran Koul
- **Student ID:** 700809605
- **Course:** Neural Network & Deep Learning (CS-5720-11833)
- **Assignment:** Part II — Programming
- **Date:** September 11, 2026

## Overview
# Tensor Manipulations & Reshaping

This program demonstrates fundamental TensorFlow tensor operations:

1. Creating a random tensor with shape `(4, 6)`.
2. Finding and printing its rank and shape using TensorFlow functions.
3. Reshaping the tensor from `(4, 6)` to `(2, 3, 4)`.
4. Transposing the reshaped tensor to `(3, 2, 4)`.
5. Broadcasting a smaller tensor of shape `(1, 4)` so it can be added to the `(2, 3, 4)` tensor.
6. Printing the rank and shape before and after the key transformations.

## Requirements

- Python 3
- TensorFlow

Install TensorFlow if it is not already installed:

```bash
pip install tensorflow
```

## How to Run

Save the supplied code as `tensor_manipulations.py`, then run:

```bash
python tensor_manipulations.py
```

## Explanation of the Work

### 1. Create the Original Tensor

The program creates a random TensorFlow tensor with shape `(4, 6)`. It contains 24 values because `4 × 6 = 24`.

```python
original_tensor = tf.random.uniform(shape=(4, 6), minval=0, maxval=10, dtype=tf.int32)
```

### 2. Find Rank and Shape

- `tf.rank(tensor)` returns the number of dimensions, also called the rank.
- `tf.shape(tensor)` returns the size of each dimension.

For the original `(4, 6)` tensor:

- Rank = `2`
- Shape = `[4, 6]`

### 3. Reshape the Tensor

The original tensor is reshaped from `(4, 6)` to `(2, 3, 4)`:

```python
reshaped_tensor = tf.reshape(original_tensor, shape=(2, 3, 4))
```

Reshaping changes how TensorFlow organizes the values into dimensions, but it does not change the values or total number of elements. Both shapes contain 24 elements:

- `(4, 6)` = 24 elements
- `(2, 3, 4)` = 24 elements

The reshaped tensor has:

- Rank = `3`
- Shape = `[2, 3, 4]`

### 4. Transpose the Reshaped Tensor

The tensor is transposed with:

```python
transposed_tensor = tf.transpose(reshaped_tensor, perm=(1, 0, 2))
```

`perm=(1, 0, 2)` changes the order of the axes:

- New axis 0 comes from old axis 1, size `3`
- New axis 1 comes from old axis 0, size `2`
- New axis 2 stays old axis 2, size `4`

Therefore, the shape changes from `(2, 3, 4)` to `(3, 2, 4)`.

The transposed tensor has:

- Rank = `3`
- Shape = `[3, 2, 4]`

### 5. Broadcasting and Addition

The program creates a smaller tensor with shape `(1, 4)`:

```python
small_tensor = tf.constant([[10, 20, 30, 40]], dtype=tf.int32)
```

The target tensor has shape `(2, 3, 4)`. TensorFlow compares dimensions from right to left when applying broadcasting:

| Dimension | Larger tensor `(2, 3, 4)` | Smaller tensor `(1, 4)` | Result |
|---|---:|---:|---|
| Last | 4 | 4 | Matches directly |
| Middle | 3 | 1 | The size-1 dimension repeats across 3 rows |
| First | 2 | Missing | TensorFlow treats it as 1 and repeats across 2 blocks |

Conceptually, TensorFlow expands `(1, 4)` to `(2, 3, 4)` without manually copying values in the code. The row `[10, 20, 30, 40]` is added to every length-4 row of the larger tensor.

```python
result_tensor = reshaped_tensor + small_tensor
```

The resulting tensor has shape `(2, 3, 4)`.

## Expected Output Structure

Because the tensor values are random, the numeric values will be different each time the program runs. However, the rank and shapes should follow this structure:

```text
Original tensor rank: 2
Original tensor shape: [4 6]

Reshaped tensor rank: 3
Reshaped tensor shape: [2 3 4]

Transposed tensor rank: 3
Transposed tensor shape: [3 2 4]

Small tensor shape: [1 4]
Broadcast-addition result shape: [2 3 4]
```

## Conclusion

This task shows that TensorFlow can inspect tensors, reshape data without changing its element count, reorder dimensions through transposition, and perform efficient arithmetic with differently shaped tensors through broadcasting.
