
<details>
<summary><h2><b> k-Nearest Neighbors (k-NN) Classifier </b></h2></summary>
<br>

The **k-Nearest Neighbors (k-NN) algorithm** functions as a memory-based classifier that utilizes a distance matrix framework. By measuring the mathematical proximity between unlabeled test data and a known training library, the algorithm assigns labels based on the most similar examples found in the existing dataset.

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


## How it Works

The prediction process follows a 5-step logic:

1. **The Scoreboard:** A Distance Matrix is created comparing Test Images (columns) against Training Images (rows).
2. **Column Scan:** For a specific Test Image, we look only at its column to find the k smallest distances.
3. **Identify Neighbors:** We find which training rows those small distances belong to.
4. **The Mapping:** We look up the labels ($y_{train}$) for those specific rows (e.g., Train 1 = Cat).
5. **Majority Vote:** The most frequent label becomes the final prediction ($y_{pred}$).


## The Distance Matrix
This table represents the L2 distances between images. **Distance Matrix Shape: (5, 3)**.

| | Test 0 | Test 1 | Test 2 | Label ($y_{train}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Train 0** | 10.5 | 2.1 | 8.0 | Dog (0) |
| **Train 1** | **1.2** | 9.5 | 1.1 | Cat (1) |
| **Train 2** | **4.3** | 8.2 | 7.5 | Dog (0) |
| **Train 3** | 12.0 | 3.0 | 4.4 | Bird (2) |
| **Train 4** | **3.7** | 1.5 | 0.9 | Cat (1) |


## Logic Example (k=3)

If we are predicting for **Test Image 0**, we find the 3 closest neighbors in its column:

1. **Smallest Distances:** 1.2, 3.7, and 4.3.
2. **Associated Labels:** Cat (1), Cat (1), and Dog (0).
3. **The Winner:** Two "Cat" votes vs. one "Dog" vote. Prediction = 1.


## Implementation Details

### Distance Formula
The algorithm uses the Euclidean (L2) distance between the flattened vectors:
$$d(p, q) = \sqrt{\sum (p_i - q_i)^2}$$


### Handling Ties
In the event of a tie (e.g., one vote each for Dog, Cat, and Bird), the algorithm selects the label with the lowest numerical index.

## Final Output
Once every column is processed, the result is a vector of winning labels:
`y_pred = [1, 0, 1]`

</details>


<details>
<summary><h2><b> Linear Classification </b></h2></summary>
<br>

In algorithms like **k-Nearest Neighbors (k-NN)**, the computer "learns" by simply memorizing all training data. This makes training fast but prediction very slow.

**Linear Classification** takes a parametric approach. It summarizes the knowledge of thousands of training images into a set of parameters: **Weights ($W$)** and **Biases ($b$)**.

* **Training (Slow):** The computer iterates through data to adjust $W$ and minimize error.
* **Testing (Fast):** We discard the training data. To predict, we simply multiply the new image by the learned matrix $W$.


## The Architecture: Scoring
The heart of the classifier is a linear equation. We treat an image as a flat vector of numbers ($x$) and multiply it by the weight matrix ($W$).

### The Score Function
$$f(x, W, b) = Wx + b$$

* **$x$ (Input):** The flattened image vector (e.g., shape $3072 \times 1$).
* **$W$ (Weights):** The learned "templates". If we have 10 classes and 3072 pixels, this matrix is $10 \times 3072$. Each row acts as a template for one specific class.
* **$b$ (Bias):** A vector of starting bonuses for each class, independent of the image data.

**The Output:** A vector of raw class scores (logits). The higher the score, the more the classifier believes the image belongs to that class.

## The Critic: Loss Functions
During training, we need to quantify "how bad" the current weights are. This measurement is called the **Loss ($L$)**.

### A. Multiclass SVM Loss (Hinge Loss)
* **Philosophy:** "I want the correct class to score higher than incorrect classes by a fixed margin ($\Delta$). Once that margin is met, I stop caring."
* **Formula:**
    $$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$
* **Interpretation:**
    * If $Score_{correct} > Score_{wrong} + \Delta$: Loss is **0** (Happy).
    * Otherwise: Loss increases linearly.

### B. Softmax Loss (Cross-Entropy)
* **Philosophy:** "Turn scores into probabilities and maximize the probability of the correct class."
* **Formula:**
    $$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum e^{s_j}}\right)$$
* **Interpretation:** It never hits strictly zero loss; it always pushes the correct score higher towards infinity.

## Regularization
If we strictly minimize Data Loss, the model might "cheat" by memorizing noise in the training data (Overfitting). We add a penalty term $R(W)$ to the loss.

$$L_{total} = \text{Data Loss} + \lambda \cdot R(W)$$

| Type | Formula | Effect |
| :--- | :--- | :--- |
| **L2 (Ridge)** | $\sum W^2$ | **Diffuse:** Spreads weights out. Encourages using all pixels a little bit. |
| **L1 (Lasso)** | $\sum \lvert W \rvert$ | **Sparse:** Forces many weights to zero. Acts as feature selection. |


## The Teacher: Optimization
We visualize the Loss Function as a rugged mountain landscape. We are standing at the top (High Loss) and want to get to the bottom (Zero Loss).

### Gradient Descent
1.  **Calculate Gradient ($\nabla W$):** Using calculus, we find the slope of the mountain. The gradient points **uphill** (direction of increasing error).
2.  **Update Step:** We take a small step in the **opposite** direction (downhill).

    $$W_{new} = W_{old} - \eta \cdot \nabla W$$
     (Where $\eta$ is the learning rate)


## The Full Cycle: A Numerical Walkthrough
We will trace a single image through the pipeline to see the math in action.

### The Setup
* **Image ($x$):** `[1, 2]` (2 pixels)
* **Correct Label ($y$):** Class 0 (**Cat**)
* **Weights ($W$):** Randomly initialized matrix.
* **Hyperparameters:** Learning Rate $\eta=0.1$, Regularization $\lambda=0.1$.

```
Weight Matrix W                     Row Interpretation
    ┌───────────────────┐            
    │   0.5     -0.2    │   ───────>   Row 0: Cat Weights
    │                   │
    │  -0.5      0.5    │   ───────>   Row 1: Dog Weights
    └───────────────────┘
```

### Step A: Forward Pass (Compute Scores)
We calculate $Wx$.
* **Cat Score:** $(0.5 \cdot 1) + (-0.2 \cdot 2) = \mathbf{0.1}$
* **Dog Score:** $(-0.5 \cdot 1) + (0.5 \cdot 2) = \mathbf{0.5}$
* **Current Prediction:** Dog. (This is **Wrong**).

### Step B: Compute Loss (SVM + Reg)
1.  **Data Loss:** We want Cat > Dog + 1.
    * Violation: $0.5 (\text{Dog}) - 0.1 (\text{Cat}) + 1 = 1.4$.
2.  **Reg Loss:** $0.1 \times \sum W^2$.
    * $0.1 \times (0.25 + 0.04 + 0.25 + 0.25) \approx 0.08$.
3.  **Total Loss:** $1.48$ (High Error).

### Step C: Backward Pass (Compute Gradient)
We determine exactly how to change the weights to fix the error. We calculate $\nabla W$.

**1. Data Gradient Logic**
The loss formula for this error is: $L = (W_{dog} \cdot x) - (W_{cat} \cdot x) + 1$.
* **Incorrect Class (Dog):** The term is positive $(+ W_{dog} \cdot x)$. The derivative is **$+x$**.
    * *Intuition:* Increasing Dog weights increases the wrong score $\rightarrow$ increases Error. Slope is positive (Uphill).
    * $\nabla_{data\_dog} = [1, 2]$
* **Correct Class (Cat):** The term is negative $(- W_{cat} \cdot x)$. The derivative is **$-x$**.
    * *Intuition:* Increasing Cat weights increases the correct score $\rightarrow$ decreases Error. Slope is negative (Downhill).
    * $\nabla_{data\_cat} = [-1, -2]$

**2. Regularization Gradient Logic**
The derivative of $\lambda W^2$ is $2\lambda W$.
* $\nabla_{reg} = 0.2 \times W$.
    * Cat Row: $[0.1, -0.04]$
    * Dog Row: $[-0.1, 0.1]$

**3. Total Gradient**
Sum Data and Reg gradients.
* **Cat Row:** $[-1, -2] + [0.1, -0.04] = \mathbf{[-0.9, -2.04]}$
* **Dog Row:** $[1, 2] + [-0.1, 0.1] = \mathbf{[0.9, 2.1]}$

### Step D: Update (Learn)
We nudge weights in the **opposite** direction of the gradient ($W_{new} = W - \eta \nabla W$).

* **New Cat Weights:**
    $$[0.5, -0.2] - (0.1 \times [-0.9, -2.04])$$
    $$= [0.5, -0.2] - [-0.09, -0.204]$$
    $$= \mathbf{[0.59, 0.004]}$$
    *(Result: Weights increased. We are pushing the Cat score UP.)*

* **New Dog Weights:**
    $$[-0.5, 0.5] - (0.1 \times [0.9, 2.1])$$
    $$= [-0.5, 0.5] - [0.09, 0.21]$$
    $$= \mathbf{[-0.59, 0.29]}$$
    *(Result: Weights decreased. We are pushing the Dog score DOWN.)*

### Final Verification
Re-calculating scores with $W_{new}$:
* **Cat Score:** $0.59$
* **Dog Score:** $-0.01$
* **Winner:** Cat.

**Conclusion:** The Backward Pass successfully identified the direction of steepest descent, and the Update Step corrected the model's weights to classify the image correctly.

</details>
