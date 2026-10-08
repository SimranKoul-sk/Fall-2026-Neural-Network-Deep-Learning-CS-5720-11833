# CS5720 Neural Network and Deep Learning: Home Assignment 3

## Student Information

- **Name:** Simran Koul
- **Student ID:** 700809605
- **Course:** CS5720 Neural Network and Deep Learning (CS-5720-11833)
- **Semester:** Fall 2026
- **Assignment:** Home Assignment 3, Part II (Programming)
- **University:** University of Central Missouri, Department of Computer Science & Cybersecurity

---

## Overview

This branch contains the Part II programming solutions for Home Assignment 3.
Each question is a standalone, commented Python script in the
[`Home-Assignment-3/`](Home-Assignment-3/) folder.

| File | Question | Topic |
| --- | --- | --- |
| [`q1_convolution_from_scratch.py`](Home-Assignment-3/q1_convolution_from_scratch.py) | Q1 | 2-D convolution implemented without a built-in convolution function |
| [`q2_transfer_learning.py`](Home-Assignment-3/q2_transfer_learning.py) | Q2 | Transfer learning with ResNet50: frozen feature extraction vs fine-tuning |

---

## Requirements

```bash
pip install numpy tensorflow matplotlib
```

Question 1 needs only NumPy. Question 2 downloads the pretrained ResNet50
weights (about 95 MB) and the `flower_photos` dataset (about 218 MB) on first
run, so it needs an internet connection the first time. Both are cached
afterwards.

## How to Run

The scripts live in `Home-Assignment-3/`, so run them from there:

```bash
cd Home-Assignment-3

python q1_convolution_from_scratch.py
python q2_transfer_learning.py
```

Question 1 finishes instantly. Question 2 trains two models and accepts an
`EPOCHS` environment variable for a quicker run:

```bash
EPOCHS=1 python q2_transfer_learning.py
```

---

## Question 1: Implement Convolution from Scratch

### Input

```text
Input matrix (5x5)        Filter (3x3)
 1  1  1  0  0             1  0  1
 0  1  1  1  0             0  1  0
 0  0  1  1  1             1  0  1
 0  0  1  1  0
 0  1  1  0  0
```

Stride = 1, padding = 0.

### Approach

No built-in convolution function is used. `convolve2d()` does the work with two
plain Python loops:

1. **(a) NumPy storage** - the input and filter are `np.array` objects built by
   `build_input_matrix()` and `build_filter()`.
2. **(b) Sliding the filter** - the two loops walk the top-left corner of the
   window across the input. The loop counters are multiplied by the stride,
   which is what makes a larger stride skip positions.
3. **(c) Dot product** - at each location the patch sitting under the filter is
   sliced out and combined with `np.sum(region * kernel)`, an element-by-element
   multiply followed by a sum.
4. **(d) and (e)** - the feature map and its shape are printed.

The output size comes from the standard formula, with integer division
discarding any position where the filter would hang off the edge:

```text
output_size = (N - F + 2P) / S + 1
```

### Output Feature Map

**Stride = 1, Padding = 0** -> shape **3 x 3**

```text
  4   3   4
  2   4   3
  2   3   4
```

Worked example for the top-left value: the window covers

```text
1 1 1        1 0 1
0 1 1   .*   0 1 0   ->  (1+0+1) + (0+1+0) + (0+0+1)  =  4
0 0 1        1 0 1
```

This result was cross-checked against `tf.nn.conv2d` and matches exactly. The
check is valid here because this particular filter is symmetric under a
180-degree rotation, so true convolution and the cross-correlation that
frameworks actually compute give the same answer.

### (f) Effect of changing the stride from 1 to 2

The script also computes the stride 2 result so the explanation can point at
real numbers:

**Stride = 2, Padding = 0** -> shape **2 x 2**

```text
  4   4
  2   4
```

The stride is how far the filter moves between positions. At stride 1 the
window shifts one column at a time and visits every valid location, giving a
3 x 3 output. At stride 2 it skips every other position, starting only at rows
and columns 0 and 2, giving a 2 x 2 output. The same formula predicts both:
`(5 - 3) / 1 + 1 = 3` and `(5 - 3) / 2 + 1 = 2`.

A larger stride downsamples the feature map. That cuts computation and memory
in the following layer and widens the receptive field of later layers, at the
cost of spatial detail, since positions the filter skipped are never measured.

Worth noticing in the two outputs above: the stride 2 map is not a summary or
an average of the stride 1 map, it is a literal **subset** of it. Every value
in the 2 x 2 result (4, 4, 2, 4) appears in the 3 x 3 result at the corners
that both strides happened to visit. Striding discards measurements rather than
combining them, which is exactly what distinguishes it from pooling.

---
