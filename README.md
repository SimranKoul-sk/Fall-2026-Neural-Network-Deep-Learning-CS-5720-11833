# CS5720 Neural Network and Deep Learning — Home Assignment 2

## Student Information

- **Name:** Simran Koul
- **Student ID:** 700809605
- **Course:** CS5720 Neural Network and Deep Learning (CS-5720-11833)
- **Semester:** Fall 2026
- **Assignment:** Home Assignment 2
- **University:** University of Central Missouri — Department of Computer Science & Cybersecurity

---

## Overview

This branch contains the solutions for Home Assignment 2, covering recurrent
neural networks (Questions 1–2) and convolutional neural networks
(Questions 3–5). Each question is a standalone, fully commented Python script
that can be run on its own. All five live in the
[`Home-Assignment-2/`](Home-Assignment-2/) folder.

| File | Question | Topic |
| --- | --- | --- |
| [`q1_rnn_text_generation.py`](Home-Assignment-2/q1_rnn_text_generation.py) | Q1 | Character-level LSTM text generation + temperature scaling |
| [`q2_sentiment_rnn_imdb.py`](Home-Assignment-2/q2_sentiment_rnn_imdb.py) | Q2 | LSTM sentiment classifier on IMDB + confusion matrix |
| [`q3_convolution_operations.py`](Home-Assignment-2/q3_convolution_operations.py) | Q3 | Convolution with different strides and padding |
| [`q4_cnn_feature_extraction.py`](Home-Assignment-2/q4_cnn_feature_extraction.py) | Q4 | Sobel edge detection + max/average pooling |
| [`q5_cnn_architectures.py`](Home-Assignment-2/q5_cnn_architectures.py) | Q5 | Simplified AlexNet + ResNet with residual blocks |

---

## Requirements

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn
```

Questions 1 and 2 download their datasets (Shakespeare text and IMDB reviews)
on first run, so an internet connection is needed the first time.

## How to Run

The scripts live in `Home-Assignment-2/`, so run them from there:

```bash
cd Home-Assignment-2

python q1_rnn_text_generation.py
python q2_sentiment_rnn_imdb.py
python q3_convolution_operations.py
python q4_cnn_feature_extraction.py
python q5_cnn_architectures.py
```

Questions 3, 4, and 5 finish in seconds. Questions 1 and 2 involve training:

- **Q1** — measured at about 133 seconds for one epoch plus text generation on
  a CPU, so the 10-epoch default works out to roughly 15–20 minutes.
- **Q2** — a few minutes; early stopping typically halts it after 2–3 epochs.

Both accept an `EPOCHS` environment variable for a quicker run:

```bash
EPOCHS=1 python q2_sentiment_rnn_imdb.py           # quick check
EPOCHS=1 TEXT_CHARS=20000 python q1_rnn_text_generation.py  # quick check
```

Question 4 optionally accepts an image path; without one it generates its own
sample image so it always runs:

```bash
python q4_cnn_feature_extraction.py my_photo.jpg
```

---

## Question 1: Implementing an RNN for Text Generation

### Approach

The script trains a **character-level** LSTM: it reads 100 characters and
predicts the single character that comes next.

1. **Dataset** — the TensorFlow Shakespeare corpus (~1.1M characters), of which
   the first 300,000 characters are used by default to keep CPU training time
   reasonable.
2. **Character encoding** — the text is reduced to its set of unique characters
   (~65 of them), and each is mapped to an integer index. Training examples are
   cut with a stride of 3 so they overlap, which yields roughly 100,000
   sequences from 300,000 characters.

   Integer indices feed an `Embedding` layer rather than one-hot vectors. Both
   were allowed by the assignment; embeddings are used because a one-hot matrix
   of shape (100,000 × 100 × 65) would waste a large amount of memory, and a
   learned dense vector per character generally trains better.
3. **Model** — `Embedding(65, 256)` → `LSTM(256)` → `Dropout(0.2)` →
   `Dense(65)`. The final layer deliberately has **no softmax activation**; it
   emits raw logits, which makes temperature scaling straightforward and
   numerically stable. The loss is configured with `from_logits=True` to match.
4. **Generation** — starting from a 100-character seed, the model predicts one
   character at a time. Each predicted character is appended to the output and
   slides into the input window, so the model continues from its own output.

### Role of Temperature Scaling

At each step the model produces one logit per character. Temperature `T`
divides those logits before the softmax:

```text
P(character) = softmax(logits / T)
```

This reshapes how peaked the probability distribution is, **without changing
anything the model learned**:

| Temperature | Effect on the distribution | Effect on the text |
| --- | --- | --- |
| **Low (0.2)** | Dividing by a small number pushes the logits further apart, sharpening the softmax onto the single most likely character | Conservative and repetitive, often looping on common words, but spelling and structure are usually correct |
| **1.0** | Logits used exactly as produced; sampling matches the learned distribution | The natural balance of variety and correctness |
| **High (1.5)** | Dividing by a larger number pulls the logits together, flattening the distribution toward uniform | Creative and surprising, but drifts into misspellings and invented words |

The core idea is that temperature **trades coherence against diversity**. As
`T → 0` sampling becomes greedy (always the top character, fully
deterministic); as `T` grows large it approaches uniform random character
choice. The script prints generated samples at T = 0.2, 0.5, 1.0, and 1.5 so
the progression is visible side by side.

### Results

After 10 epochs the model reached a training loss of **1.3774** and training
accuracy of **0.5781** — meaning it predicts the correct next character about
58% of the time. Seeded with the opening 100 characters of the corpus, the four
temperatures produce visibly different text:

**T = 0.2** — correct spelling and believable structure, but stuck in a loop:

```text
MENENIUS:
The lord of the commind and better are
The people the comming than the people of the people,
That shall be convers the people of the commandant.
```

**T = 0.5** — more varied vocabulary, occasional malformed words:

```text
BRUTUS:
The gatest to do your grace some fall of mouth
That was all the censents of a god of my prayor words,
```

**T = 1.0** — inventive and far less repetitive, but many words are made up:

```text
CORIOLANUS:
Dich is rememplabineds, and that people
mnouther. He wolds no metwors;
```

**T = 1.5** — mostly nonsense words, though the play-script shape survives:

```text
COMINIUS:
'Tis? Whe, I wages mady?
You knop foono'llokips: it,
```

The progression matches the theory above. What is worth noticing is that even
at T = 1.5 the model still produces capitalised speaker names followed by a
colon and a line break — the format is learned so strongly that it survives
heavy random sampling, long after individual words have stopped being words.

---

## Question 2: Sentiment Classification Using RNN

### Approach

1. **Dataset** — `tensorflow.keras.datasets.imdb`, 25,000 training and 25,000
   test movie reviews, balanced 50/50 positive and negative. Keras ships it
   pre-tokenized, with each word replaced by its frequency rank.
2. **Preprocessing** — `num_words=10000` keeps only the 10,000 most common
   words (rarer ones become an out-of-vocabulary token), which bounds the
   embedding size. Reviews vary from 11 to 2,494 tokens, but an LSTM batch
   needs a rectangular array, so `pad_sequences` trims and zero-pads every
   review to 200 tokens.
3. **Model** — `Embedding(10000, 128, mask_zero=True)` → `LSTM(64)` →
   `Dropout(0.5)` → `Dense(1, sigmoid)`.

   `mask_zero=True` matters: it tells the LSTM to skip the padding tokens, so a
   short review padded out to 200 tokens is not diluted by 180 meaningless
   zeros. The LSTM reads the review in order and carries context forward, which
   is what lets it handle negation such as *"not good"* — a bag-of-words model
   would see the word "good" and get it backwards.
4. **Training** — up to 5 epochs with an `EarlyStopping` callback watching
   validation loss (`patience=1`, `restore_best_weights=True`). This was added
   after measuring the problem directly: without it, training for a fixed 3
   epochs gave **83.18%** accuracy at a test loss of **0.5349**, while a single
   epoch gave essentially the same accuracy (83.20%) at a much better loss of
   **0.4038**. The model was overfitting after epoch 1 — training accuracy kept
   climbing while the model got worse at generalizing. Early stopping restores
   the best epoch's weights instead of whichever happened to be last.
5. **Evaluation** — the sigmoid output is thresholded at 0.5 to produce class
   labels, then scored with `confusion_matrix` and `classification_report` from
   scikit-learn. The confusion matrix is also saved as
   `q2_confusion_matrix.png`.

### Results

Early stopping halted training at epoch 3 and restored the weights from
epoch 2, giving **test accuracy 0.8493** at a test loss of **0.3633** — better
on both counts than the fixed 3-epoch run it replaced.

```text
Confusion Matrix
                    Predicted Negative   Predicted Positive
Actual Negative           10,759               1,741
Actual Positive            2,027              10,473

Classification Report
              precision    recall  f1-score   support
    Negative     0.8415    0.8607    0.8510     12500
    Positive     0.8575    0.8378    0.8475     12500
    accuracy                         0.8493     25000
```

These exact numbers will vary slightly between runs, since dropout and the
weight initialization are stochastic.

### Why the Precision-Recall Tradeoff Matters

The two metrics answer different questions about the positive class:

```text
Precision = TP / (TP + FP)   Of the reviews we called positive, how many really were?
Recall    = TP / (TP + FN)   Of the reviews that really were positive, how many did we find?
```

The run above shows this concretely. For the positive class, precision is
**0.8575** but recall is only **0.8378** — the model is somewhat reluctant to
call a review positive, so when it does say positive it is usually right
(few false positives: 1,741), but it misses more genuinely positive reviews
(2,027 false negatives). The negative class shows the mirror image. Accuracy
alone (0.8493) hides that asymmetry completely.

They trade off against each other because both are governed by the **same
decision threshold** on the sigmoid output. Lowering the threshold below 0.5
labels more reviews positive, catching more true positives (recall up) while
letting more negatives slip through (precision down). Raising it does the
reverse. A single trained model therefore offers a whole curve of
precision/recall pairs, not one fixed operating point.

Which side to favour depends entirely on what each error costs:

- **Brand monitoring**, where negative reviews are escalated to a support team:
  missing a genuinely angry customer is expensive, so **recall** on the negative
  class matters most. A few false alarms only cost a moment of a human's time.
- **Pulling positive quotes for marketing**: a single negative review
  mistakenly quoted as praise is public and embarrassing, so **precision**
  matters far more. Missing some good reviews is harmless — there are plenty.

This is also why accuracy alone is insufficient. IMDB is balanced, but on a
skewed real-world feed where 95% of reviews are positive, a model that blindly
predicts "positive" scores 95% accuracy while catching *zero* negative reviews.
The confusion matrix and F1-score (the harmonic mean of precision and recall)
expose that failure; accuracy hides it.

---

## Question 3: Convolution Operations with Different Parameters

### Input

```text
Input matrix (5x5)              Kernel (3x3)
  1   2   3   4   5               0   1   0
  6   7   8   9  10               1  -4   1
 11  12  13  14  15               0   1   0
 16  17  18  19  20
 21  22  23  24  25
```

The convolutions are performed with `tf.nn.conv2d`, which requires 4D tensors,
so the input is reshaped to `(1, 5, 5, 1)` — `(batch, height, width, channels)` —
and the kernel to `(3, 3, 1, 1)` — `(k_height, k_width, in_channels, out_channels)`.

> **Note on convolution vs. cross-correlation:** `tf.nn.conv2d` technically
> computes cross-correlation (it does not flip the kernel). This particular
> kernel is symmetric under a 180° rotation, so for this filter the two
> operations give identical results.

### Output Feature Maps

**Stride = 1, Padding = 'VALID'** → shape 3 × 3

```text
  0.0   0.0   0.0
  0.0   0.0   0.0
  0.0   0.0   0.0
```

**Stride = 1, Padding = 'SAME'** → shape 5 × 5

```text
   4.0    3.0    2.0    1.0   -6.0
  -5.0    0.0    0.0    0.0  -11.0
 -10.0    0.0    0.0    0.0  -16.0
 -15.0    0.0    0.0    0.0  -21.0
 -46.0  -27.0  -28.0  -29.0  -56.0
```

**Stride = 2, Padding = 'VALID'** → shape 2 × 2

```text
  0.0   0.0
  0.0   0.0
```

**Stride = 2, Padding = 'SAME'** → shape 3 × 3

```text
   4.0    2.0   -6.0
 -10.0    0.0  -16.0
 -46.0  -28.0  -56.0
```

### Why the all-zero outputs are correct, not a bug

The VALID results being entirely zero looks alarming, but it is the
mathematically correct answer. The kernel `[[0,1,0],[1,-4,1],[0,1,0]]` is a
**discrete Laplacian**, which measures how much a pixel differs from the
average of its four neighbours. The input matrix is a perfectly linear ramp
(value = 5·row + col + 1), and every pixel in it is *exactly* the average of its
four neighbours. The Laplacian of any linear function is zero, so the interior
response is zero everywhere.

The non-zero values in the SAME results appear only along the borders, where
the zero-padding breaks that linearity — the kernel there is comparing real
values against padded zeros.

### Why the output sizes differ

| Stride | Padding | Output | Reason |
| --- | --- | --- | --- |
| 1 | VALID | 3 × 3 | No padding, so the 3×3 kernel only sits where it fully fits: `5 − 3 + 1 = 3` |
| 1 | SAME | 5 × 5 | Zeros are added around the border so the output matches the input size |
| 2 | VALID | 2 × 2 | Kernel skips every other position: `⌈(5 − 3 + 1) / 2⌉ = 2` |
| 2 | SAME | 3 × 3 | Padded, then halved by the stride: `⌈5 / 2⌉ = 3` |

---

## Question 4: CNN Feature Extraction with Filters and Pooling

### Task 1: Edge Detection Using Sobel Filters

The script applies the two Sobel kernels from the assignment with
`cv2.filter2D`, using the exact kernels given rather than OpenCV's built-in
`cv2.Sobel()`:

```text
Sobel X (vertical edges)      Sobel Y (horizontal edges)
 -1   0   1                    -1  -2  -1
 -2   0   2                     0   0   0
 -1   0   1                     1   2   1
```

Two implementation details worth noting:

- **`ddepth=cv2.CV_64F`** — edge responses are *signed*: a dark-to-light edge is
  positive and light-to-dark is negative. Storing the result as `uint8` would
  clip every negative value to zero and silently lose half the edges. The
  results are passed through `np.absolute()` only at display time.
- If no image path is given on the command line, the script generates a
  200×200 sample image containing a rectangle, a circle, and a diagonal line —
  shapes chosen so the difference between the two filters is obvious.

The output (`q4_sobel_edge_detection.png`) shows the expected behaviour clearly:
**Sobel-X** responds to the rectangle's *vertical* sides and the circle's left
and right arcs, while **Sobel-Y** responds to the *horizontal* sides and the
circle's top and bottom arcs. The diagonal line appears in both, since it has
gradient in both directions.

### Task 2: Max Pooling and Average Pooling

A random 4×4 matrix (seeded with 42 for reproducibility) is passed through
`MaxPooling2D(pool_size=(2,2))` and `AveragePooling2D(pool_size=(2,2))`. Keras
pooling layers default to a stride equal to the pool size, so the 2×2 windows do
not overlap and a 4×4 input becomes 2×2.

```text
Original 4x4          Max-pooled 2x2      Average-pooled 2x2
 6   3   7   4          9    7              6.00   4.75
 6   9   2   6          7    7              5.00   4.75
 7   4   3   7
 7   2   5   4
```

Working through the top-left 2×2 window `[[6,3],[6,9]]`: max pooling keeps `9`,
average pooling keeps `(6+3+6+9)/4 = 6.0`.

**Max pooling** keeps the single strongest activation in each window, preserving
the most prominent detected feature and discarding the rest — which is why it
dominates in practice for CNNs. **Average pooling** keeps the mean, which
smooths the response and retains more background information but dilutes sharp
features.

---

## Question 5: Implementing and Comparing CNN Architectures

### Task 1: Simplified AlexNet

Built with the Sequential API on a 227×227×3 input (the size used in the
original AlexNet paper, which works cleanly with an 11×11 kernel at stride 4).

| Layer | Output Shape | Params |
| --- | --- | --- |
| Conv2D (96, 11×11, stride 4, ReLU) | (55, 55, 96) | 34,944 |
| MaxPooling2D (3×3, stride 2) | (27, 27, 96) | 0 |
| Conv2D (256, 5×5, ReLU) | (23, 23, 256) | 614,656 |
| MaxPooling2D (3×3, stride 2) | (11, 11, 256) | 0 |
| Conv2D (384, 3×3, ReLU) | (9, 9, 384) | 885,120 |
| Conv2D (384, 3×3, ReLU) | (7, 7, 384) | 1,327,488 |
| Conv2D (256, 3×3, ReLU) | (5, 5, 256) | 884,992 |
| MaxPooling2D (3×3, stride 2) | (2, 2, 256) | 0 |
| Flatten | (1024,) | 0 |
| Dense (4096, ReLU) | (4096,) | 4,198,400 |
| Dropout (0.5) | (4096,) | 0 |
| Dense (4096, ReLU) | (4096,) | 16,781,312 |
| Dropout (0.5) | (4096,) | 0 |
| Dense (10, Softmax) | (10,) | 40,970 |

**Total parameters: 24,767,882**

**Padding note:** every convolution layer uses Keras's default `padding='valid'`,
since the assignment specifies a kernel size and (where relevant) a stride but
no padding. Each convolution therefore loses a border of pixels, which is why
the feature map shrinks steadily from 55×55 down to 2×2 before the flatten.

Two observations from the summary:

- The pooling layers use a 3×3 window with stride 2, so the windows **overlap**.
  This was one of AlexNet's contributions over earlier non-overlapping pooling.
- **85% of the parameters sit in the fully connected layers** (21.0M of 24.8M),
  while all five convolution layers together account for only 3.7M. This is
  exactly why the 50% dropout layers are placed on the dense layers — they are by
  far the most prone to overfitting.

### Task 2: Residual Block and ResNet

The residual block is implemented with the functional API, since a skip
connection is not a straight line of layers:

```text
input ──> Conv2D(64, 3x3, ReLU) ──> Conv2D(64, 3x3, linear) ──> Add ──> ReLU ──> output
     │                                                           ▲
     └───────────────────── skip connection ─────────────────────┘
```

Two deliberate choices:

- The second convolution is left **linear**, with the ReLU applied *after* the
  addition. This is the standard ordering from the ResNet paper, and it matches
  the assignment's wording that the skip connection is added "before activation".
- `padding='same'` is **required** inside the block, not optional: the block's
  output must keep the same height, width, and channel count as its input, or the
  `Add` layer could not combine the two tensors.

The full model on a 64×64×3 input:

| Layer | Output Shape | Params |
| --- | --- | --- |
| Input | (64, 64, 3) | 0 |
| Conv2D (64, 7×7, stride 2) | (32, 32, 64) | 9,472 |
| Residual block 1 (2 × Conv2D + Add + ReLU) | (32, 32, 64) | 73,856 |
| Residual block 2 (2 × Conv2D + Add + ReLU) | (32, 32, 64) | 73,856 |
| Flatten | (65536,) | 0 |
| Dense (128, ReLU) | (128,) | 8,388,736 |
| Dense (10, Softmax) | (10,) | 1,290 |

**Total parameters: 8,547,210**

### Comparison

**AlexNet** is a plain stack — every layer feeds only the next one. Its depth is
limited by the vanishing-gradient problem: the further back a gradient travels
through a plain stack, the more it shrinks, until the early layers barely learn.

**ResNet** adds skip connections, so each block only has to learn the *residual*
(the change from its input) rather than an entire transformation. If a block is
not useful, it can learn to output roughly zero and pass its input through
unchanged. Crucially, the `Add` gives gradients a direct path backwards that
bypasses the convolutions entirely, which is what allows ResNets to reach
hundreds of layers where plain stacks like AlexNet degrade past about 20.

Note also where the parameters live: AlexNet's 24.8M is dominated by dense
layers, and the small ResNet's 8.5M is likewise dominated by the single
`Dense(128)` after the flatten (8.39M of 8.55M). Real ResNets replace that
flatten with global average pooling, which removes almost all of those
parameters.

---

## Output Files

Running the scripts produces these inside `Home-Assignment-2/`:

- `q1_char_lstm.keras` — the trained text-generation model, saved so generation
  can be re-run without retraining (not committed; see `.gitignore`)
- `q2_confusion_matrix.png` — labelled confusion matrix for the IMDB classifier
- `q4_sobel_edge_detection.png` — original image alongside the Sobel-X and
  Sobel-Y results
