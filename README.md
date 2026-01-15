# k-Nearest Neighbors (k-NN) Classifier

This project implements a k-Nearest Neighbors algorithm using a distance matrix approach. It predicts the labels of test images by identifying the closest examples from a training dataset.



## Image Data and Pixel Shaping

Before calculating distances, raw image data must be transformed into a format the computer can process.

### 1. From Grid to Vector (Flattening)
A standard image is a grid of pixels. To compare images mathematically, we "unroll" or "flatten" the grid into a single row of numbers (a vector).

**Visual Example: 2x2 Grayscale Image**

```text
STEP 1: Original Grid        STEP 2: Read Rows        STEP 3: Flattened Vector
    ┌───────────┐                                      
    │  255    0 │  ───────>  Row 1: [255, 0]  ───────>  [255, 0, 128, 50]
    │  128   50 │  ───────>  Row 2: [128, 50]
    └───────────┘
```

### 2. Toy Dataset Dimensions
When we have multiple images, we stack these vectors. In this example, we have 5 Training images and 3 Test images, where each image is a 2x2 grid (4 total pixels).

**Training Data ($X_{train}$)**
*Shape: (5, 4)*
| Image ID | P1 | P2 | P3 | P4 |
| :--- | :--- | :--- | :--- | :--- |
| Train 0 | 255 | 0 | 128 | 50 |
| Train 1 | 10 | 10 | 0 | 255 |
| Train 2 | ... | ... | ... | ... |

**Test Data ($X_{test}$)**
*Shape: (3, 4)*
| Image ID | P1 | P2 | P3 | P4 |
| :--- | :--- | :--- | :--- | :--- |
| Test 0 | 250 | 5 | 130 | 45 |
| Test 1 | ... | ... | ... | ... |

---

## How it Works

The prediction process follows a 5-step logic:

1. **The Scoreboard:** A Distance Matrix is created comparing Test Images (columns) against Training Images (rows).
2. **Column Scan:** For a specific Test Image, we look only at its column to find the k smallest distances.
3. **Identify Neighbors:** We find which training rows those small distances belong to.
4. **The Mapping:** We look up the labels ($y_{train}$) for those specific rows (e.g., Train 1 = Cat).
5. **Majority Vote:** The most frequent label becomes the final prediction ($y_{pred}$).

---

## The Distance Matrix
This table represents the L2 distances between images. **Distance Matrix Shape: (5, 3)**.

| | Test 0 | Test 1 | Test 2 | Label ($y_{train}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Train 0** | 10.5 | 2.1 | 8.0 | Dog (0) |
| **Train 1** | **1.2** | 9.5 | 1.1 | Cat (1) |
| **Train 2** | **4.3** | 8.2 | 7.5 | Dog (0) |
| **Train 3** | 12.0 | 3.0 | 4.4 | Bird (2) |
| **Train 4** | **3.7** | 1.5 | 0.9 | Cat (1) |

---

## Logic Example (k=3)

If we are predicting for **Test Image 0**, we find the 3 closest neighbors in its column:

1. **Smallest Distances:** 1.2, 3.7, and 4.3.
2. **Associated Labels:** Cat (1), Cat (1), and Dog (0).
3. **The Winner:** Two "Cat" votes vs. one "Dog" vote. Prediction = 1.

---

## Implementation Details

### Distance Formula
The algorithm uses the Euclidean (L2) distance between the flattened vectors:
$$d(p, q) = \sqrt{\sum (p_i - q_i)^2}$$



### Handling Ties
In the event of a tie (e.g., one vote each for Dog, Cat, and Bird), the algorithm selects the label with the lowest numerical index.

## Final Output
Once every column is processed, the result is a vector of winning labels:
`y_pred = [1, 0, 1]`
