
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
* **Interpretation:** It never hits strictly zero loss; it always pushes the correct score higher towards infinity.z

### C. The Final Objective (Data Loss)
* **Philosophy:** "Minimize the average error across the entire dataset to find the best fit."
* **Formula:**
    $$L = \frac{1}{N} \sum_{i=1}^{N} L_i$$
* **Interpretation:**
    * **$N$:** The total number of images in the training batch.
    * **$\sum L_i$:** The sum of losses for every individual image (calculated using method A or B).
    * We optimize the **average** loss so the model learns patterns common to all data, rather than just fixing one specific image.

> **Note:** To improve performance and speed, we don't look at the entire dataset at once. Instead, we compute this average using a **small batch of images** (e.g., 32 or 64). This method is called **Stochastic Gradient Descent (SGD)**.

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
    * *Math:* $\frac{\partial L}{\partial W_{dog}} = x$
    * $\nabla_{data\_{dog}} = [1, 2]$
* **Correct Class (Cat):** The term is negative $(- W_{cat} \cdot x)$. The derivative is **$-x$**.
    * *Intuition:* Increasing Cat weights increases the correct score $\rightarrow$ decreases Error. Slope is negative (Downhill).
    * *Math:* $\frac{\partial L}{\partial W_{cat}} = -x$
    * $\nabla_{data\_{cat}} = [-1, -2]$

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
In the final step, we perform **Gradient Descent**. We "learn" by nudging the weights in the direction that reduces the error. We nudge weights in the **opposite** direction of the gradient ($W_{new} = W - \eta \nabla W$).

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

<details>
<summary><h2><b> Neural Networks (Multi-Layer Perceptrons) </b></h2></summary>
<br>

Linear Classifiers are limited: they learn a single "template" per class. If a problem is not linearly separable (like an XOR puzzle or a donut shape), a Linear Classifier fails completely.

**Neural Networks** overcome this by stacking linear layers on top of each other, separated by non-linear functions. This allows the computer to warp the input space to classify complex, hierarchical patterns.

## The Architecture: Layers and Neurons

We treat the data as a signal flowing through a sequence of gates. The Activation Function must come **after** the linear summation to create the feature.

```text
      Input (x)                  Hidden Layer Steps (h)                       Output (s)
    ┌───────────┐           ┌───────────────────┬──────────────┐            ┌────────────┐
      Raw Data    ── W1 ──▶  1. Linear Sum (z)   2. Activation   ── W2 ──▶  Class Score 
    └───────────┘           └───────────────────┴──────────────┘            └────────────┘
                                      │                ▲
                                      ▼                │
                                  Result = Wx+b    Apply ReLU (f)
```

### 1. The Hidden Layer
The "middle" of the sandwich. It takes raw inputs and transforms them into features.
* **Formula:** $h = f(W_1x + b_1)$
* **Interpretation:** The network creates its own internal understanding of the image before trying to classify it.

### 2. The Activation Function ($f$)

This is the most critical component. If we simply stacked linear layers without this, $W_2(W_1x)$ is just $W_{new}x$. The network would collapse back into a Linear Classifier. We need a "non-linearity" to break the line.

**Common Choice: ReLU (Rectified Linear Unit)**
* **Formula:** $f(x) = \max(0, x)$
* **Philosophy:** "If the signal is positive, pass it through. If it's negative, kill it (set to zero)." It acts like a gatekeeper.

### 3. The Output Layer
Takes the hidden features ($h$) and produces the final class scores.
* **Formula:** $s = W_2h + b_2$

## The Teacher: Backpropagation (The Chain Rule)

In a simple Linear Classifier, changing weights has a direct, obvious effect on the score. In a Neural Network, the weights in the early layers ($W_1$) are buried deep inside. They are like workers at the beginning of an assembly line—they don't see the final product.

To fix mistakes, we use **Backpropagation** (The Chain Rule). Think of it as a "Blame Game."

**The Logic:**
1.  We calculate the total error (Loss) at the very end.
2.  We ask the final layer: *"How much did you contribute to this error?"*
3.  We pass that blame backward to the previous layer and ask: *"And how much was YOUR fault?"*

### The Flow of Information
We move forward to make predictions, and backward to learn.

```text
            FORWARD PASS (Make Prediction) ───────────────▶
    
      Input (x)            Hidden (h)           Output (s)          Loss (L)
    ┌───────────┐        ┌────────────┐       ┌────────────┐      ┌──────────┐
      Raw Data   ───W1──▶  Features   ───W2──▶ Raw Scores   ──▶   Calculate
    │           │        │            │       │            │      │  Error   │
    └───────────┘        └────────────┘       └────────────┘      └─────┬────┘
          ▲                    ▲                    ▲                   │
          │                    │                    │                   │
          └────────────────────┴─── BACKWARD PASS ──┴───────────────────┘
                                   (Assign Blame)
          
     "Hey W1, your features    "Hey W2, your scores    "Hey W2, the Loss is
      contributed to the       were off! Change how     high! Adjust yourself
      error W2 reported."      you weight h."           based on h."
```

### How to Calculate the Gradient (The Math)
At every step, we calculate the gradient using a simple logic that maps directly to Calculus (The Chain Rule):

> **Total Gradient = Incoming Blame × My Local Effect**

**The Math (Chain Rule):**

> $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

1.  **Incoming Blame (Upstream Gradient):** The error signal coming from the layer ahead of you ($\frac{\partial L}{\partial y}$).
2.  **My Local Effect (Local Gradient):** The derivative of your specific math operation ($\frac{\partial y}{\partial x}$).
3.  **Total Gradient:** You multiply them to find your total responsibility ($\frac{\partial L}{\partial x}$).

### Expanded: The Chain Rule for a Series of Layers
In a neural network, the signal passes through many layers ($x \to h \to s \to L$). To find the gradient at the start, we simply extend the chain.

**The Formula:**

> $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial s} \cdot \frac{\partial s}{\partial h} \cdot \frac{\partial h}{\partial x}$$

**How it works (The "Domino Effect"):**
* **Layer 3 (Loss):** Calculates the initial error ($\frac{\partial L}{\partial s}$).
* **Layer 2:** Multiplies that error by its own local derivative ($\frac{\partial s}{\partial h}$) and passes it back.
* **Layer 1:** Receives the combined error and multiplies by its own derivative ($\frac{\partial h}{\partial x}$).

### Practical Application: How do I fix the Weights?
To update a specific weight, we calculate the gradient **with respect to that weight**. This means we follow the chain of derivatives from the Loss backward and **stop exactly at the weight we are updating**.



**1. To fix $W_2$:**
We only need to look at the path from **Loss $\to$ Output $\to$ $W_2$**.
> $$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial s} \cdot \frac{\partial s}{\partial W_2}$$

* **Logic:** Since $s = h \cdot W_2$, the derivative is just the **Input** ($h$).
* **Result:** $\text{Incoming Blame }(\frac{\partial L}{\partial s}) \cdot \text{Input }(h)$

**2. To fix $W_1$:**
We must go deeper: **Loss $\to$ Output $\to$ Hidden ($h, z$) $\to$ $W_1$**.

> $$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial W_1}$$

* **Definition ($\frac{\partial L}{\partial z}$):** This symbol is a **shorthand** for the entire chain leading up to this node. It combines the Output Error, the Weight $W_2$, and the ReLU derivative:
    $$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial s} \cdot \frac{\partial s}{\partial h} \cdot f'(z)$$

  $$\frac{\partial L}{\partial z} = \text{Incoming Blame} \cdot W_2 \cdot f'(z)$$

  * **The Weight ($W_2$):** Scales the blame based on the connection strength ($\frac{\partial s}{\partial h}$). Since the final score is $s = h \cdot W_2$, the derivative is simply **$W_2$**.
  * **The Switch ($f'$):** Acts as a gate based on the forward pass. if $z > 0$, multiply by **1** (pass); if $z \le 0$, multiply by **0** (block).

* **Logic:** Since the local math is $z = x \cdot W_1$, the derivative $\frac{\partial z}{\partial W_1}$ is just the **Input** ($x$).
* **Result:** $\frac{\partial L}{\partial z} \cdot x$

### How We Update the Weights (Optimization)
Once we have calculated the gradient **for a specific weight**, we know which direction increases error for that weight. To learn, we move that weight in the opposite direction.

**The Formula (Gradient Descent):**
$$W_{new} = W_{old} - (\eta \cdot \text{Gradient})$$

* **$W_{old}$:** The current weight.
* **$-$ (Minus):** Walk downhill (opposite to error).
* **$\eta$ (Learning Rate):** The step size (e.g., 0.01).
* **Gradient:** The slope we just calculated.

## The Full Cycle: A Numerical Walkthrough

We will trace a single data point through a 2-Layer Network to see the Chain Rule in action.

### The Setup
We use simplified **scalars** (single numbers) instead of vectors to make the math transparent.

| Component | Value | Description |
| :--- | :--- | :--- |
| **Input ($x$)** | `2` | A single feature (e.g., Pixel Brightness). |
| **Target ($y$)** | `1` | The "Correct Answer" we want to predict. |
| **Weights ($W$)** | `3.0`, `-2.0` | **W1** (Input → Hidden), **W2** (Hidden → Output). |
| **Rate ($\eta$)** | `0.01` | The step size for Gradient Descent. |

### Step A: Forward Pass (Compute Score)
We calculate the values from left to right.

```text
    1. Input        2. Linear        3. ReLU         4. Score        5. Loss
    ┌─────┐  W1=3   ┌─────┐         ┌─────┐  W2=-2   ┌─────┐         ┌─────┐
    │  2  │──(* 3)─▶  6    ──(Max)─▶  6   ──(* -2)─▶ -12  ──(Diff)─▶ 169 
    └─────┘         └─────┘         └─────┘          └─────┘         └─────┘
       x               z               h                s               L
```

1.  **Hidden ($z$):** $x * 3.0 = 6$
2.  **Activation ($h$):** $\text{ReLU}(6) = 6$ (Gate is OPEN because $6 > 0$)
3.  **Score ($s$):** $h * -2.0 = -12$
4.  **Loss ($L$):** $(s - y)^2 = (-12 - 1)^2 = (-13)^2 = 169$

*Status: The model predicts -12. We wanted 1. Massive Error.*

### Step B: Backward Pass (Calculate Gradients)
We move from **Right to Left** to find $\nabla W_1$ and $\nabla W_2$.
**The Goal:** Minimize the Loss Formula: $L = (s - y)^2$

**Logic:** `Total Gradient = Incoming Blame * Local Effect`

```text
    Step 4           Step 3           Step 2           Step 1
   Pass Blame       Pass Blame       Pass Blame       Start Here
     ◀────            ◀────            ◀────             │
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │   104    │     │    52    │     │   -156   │     │   -26    │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘
    d(W1)/dL         d(ReLU)          d(W2)/dL          dL/ds
```

**1. Gradient at Output (Box: `dL/ds`)**
We start by deriving the Loss formula $(s - y)^2$.
* **Chain Rule:** $\frac{\partial L}{\partial s} = 2(s - y)$
* **Math:** $2 * (-12 - 1)$
* **Calc:** $\mathbf{-26}$
* *Note: Negative gradient means we need a positive push.*

**2. Gradient for W2 (Box: `d(W2)/dL`)**
* **Chain Rule:** $\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial s} \cdot h$
* **Math:** $\text{Incoming Blame} (-26) * \text{Input } (6)$
* **Calc:** $-26 * 6 = \mathbf{-156}$
* **Action:** The gradient is very negative. To fix the error, we must **increase** $W_2$ significantly.

**3. Gradient at Hidden / ReLU (Box: `d(ReLU)`)**
We need to pass the blame *through* the hidden layer (across the wire $W_2$ and through the ReLU gate).
* **Chain Rule:** $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial s} \cdot W_2 \cdot f'(z)$
* **Math:** $\text{Blame } (-26) * \text{Weight } (-2) * \text{ReLU } (1)$
* **Calc:** $-26 * -2 * 1 = \mathbf{52}$
* **Note 1 (The Weight):** We multiply by $W_2$ because it acts as the bridge for the error to travel back.
* **Note 2 (The ReLU):** We multiply by **1** because the gate was **OPEN** ($x>0$). If closed, it would be 0.
* *Why W2?* The weight is the path. Since $W_2$ multiplied the signal going forward, we must multiply by $W_2$ to send the blame back across that same path.
* *Why ReLU?* If the ReLU gate was closed, the derivative is 0 ("Neuron didn't fire, so don't blame it").

**4. Gradient for W1 (Box: `d(W1)/dL`)**
* **Chain Rule:** $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z} \cdot x$
* **Math:** $\text{Incoming Blame } (52) * \text{Input } (2)$
* **Calc:** $52 * 2 = \mathbf{104}$
* **Action:** The gradient is positive. To fix the error, we must **decrease** $W_1$ to reduce the signal strength.

### Step C: The Update (Gradient Descent)
We nudge the weights in the **opposite** direction of the gradient to reduce error.

$$W_{1_{new}} = 3.0 - (0.01 \cdot 104) = \mathbf{1.96}$$
*(We decreased W1 to reduce the magnitude of the signal at the start)*

$$W_{2_{new}} = -2.0 - (0.01 \cdot -156) = \mathbf{-0.44}$$
*(We increased W2 to stop it from flipping the score to negative)*

### Final Verification: Did it actually work?
We updated our weights to **$W_1 = 1.96$** and **$W_2 = -0.44$**.
Now, let's run the **Forward Pass again** with the *same input* ($2$) but these *new weights* to see if the network learned.

**1. The New Forward Pass**
* **Input:** $2$
* **New Hidden ($h$):** $2 * 1.96 = \mathbf{3.92}$
* **New Score ($s$):** $3.92 * -0.44 \approx \mathbf{-1.72}$

**2. The Comparison (Target = 1)**
We want the score to be as close to **1** as possible.

| State | Prediction | Distance from Target (Error) |
| :--- | :--- | :--- |
| **Before Learning** | `-12.00` | `13.00` (Huge miss) |
| **After 1 Step** | `-1.72` | `2.72` (Much closer) |

**Conclusion:**
In just one single step of Backpropagation, we moved the prediction from -12 to -1.72. We reduced the error by **~79%**.
*If we repeat this process 10-20 times (a "Training Loop"), the prediction will eventually land exactly on 1.0.*

</details>

<details>
<summary><h2><b> Convolutional Neural Networks (CNNs) </b></h2></summary>
<br>

While Multi-Layer Perceptrons (MLPs) are powerful, they have a major flaw when handling images: they destroy spatial structure. To feed an image into an MLP, you must "flatten" it (turn a 2D grid into a long line). The network loses the understanding that pixel (0,0) is next to pixel (0,1).

**Convolutional Neural Networks (CNNs)** are designed to process data with a grid-like topology (like images). Instead of looking at the whole image at once, they scan it piece-by-piece to build a hierarchy of features.

## The Intuition: The Scanning Flashlight

Imagine you are in a dark room looking for a specific object (e.g., a "horizontal edge"). You have a small flashlight.
* You don't flash the whole room at once.
* You shine the light on the top-left corner, then slide it slightly to the right, scanning the entire room row by row.
* Whenever the light hits a "horizontal edge," your detector beeps.

In a CNN, the **Flashlight** is the **Filter (or Kernel)**, and the **Beep** is the **Activation**.

## The Architecture: The Visual Cortex

A CNN is typically a pipeline of three repeating stages, followed by a classifier.

```text
      Input Image             Feature Extraction (The Eye)            Classification (The Brain)
    ┌─────────────┐      ┌───────────┬───────────┬───────────┐      ┌───────────┬────────────┐
        Raw 2D       ──▶  1. Conv      2. ReLU    3. Pool     ──▶    4. Flat      5. Dense  
    │   Pixels    │      │   Layer   │           │   Layer   │      │   Vector  │   Output   │
    └─────────────┘      └───────────┴───────────┴───────────┘      └───────────┴────────────┘
                              │            │           │                      ▲
                              ▼            ▼           ▼                      │
                        Detect Features   Gate      Shrink Image         Make Decision
```

### 1. The Convolution Layer (The Filter)
The core building block. Instead of connecting every input to every neuron (like MLPs), we use a small **Filter** (or Kernel)—e.g., a $3\times3$ grid of weights—that slides over the input.
* **Weight Sharing:** The *same* filter is used across the entire image. This means if the network learns to detect a "vertical line" in the top-left, it can detect it in the bottom-right using the same weights.
* **The Math:** It performs a **Dot Product** (element-wise multiplication followed by a sum) between the filter weights and the patch of the image it currently covers.

### 2. The Activation (ReLU)
Just like in MLPs, we need non-linearity. We apply **ReLU** `f(x) = max(0, x)` to every pixel in the feature map.
* **Purpose:** It acts as a gate. If the filter found a feature (positive value), the signal passes. If not (negative or zero), the signal is killed.

### 3. The Pooling Layer (Downsampling)
Convolution creates a precise map of *where* features are. Pooling summarizes this map to make the network robust to small movements (translation invariance) and to reduce the file size.
* **Max Pooling:** We slide a window (usually $2\times2$) and keep only the **largest** number in that window.
* **Philosophy:** "I don't need to know the pixel-perfect coordinate of the cat's ear, just that it's in this general quadrant."

## The Mechanics: Sliding Windows & Hyperparameters

To control how the filter scans the image, we adjust these knobs:

1.  **Filter Size ($F$):** The dimensions of the scanning window (usually $3\times3$ or $5\times5$).
2.  **Stride ($S$):** The step size.
    * *Stride 1:* Slide 1 pixel at a time (High overlap, high detail).
    * *Stride 2:* Skip 1 pixel every step (Reduces output size by half).
3.  **Padding ($P$):** Adding a border of "fake" pixels (usually zeros) around the input so the filter can center over the corners.

**Output Size Formula:**
> $$\text{Output Size} = \frac{W - F + 2P}{S} + 1$$
*(Where **W** is Input Width, **F** is Filter Size, **P** is Padding, **S** is Stride)*


## Visualizing the Forward Pass (5x5 Input) 

We will trace a signal from raw pixels to a final class score.

**The Setup:**
* **Input Image (X):** A **5x5** grid of pixels (Grayscale brightness 0-255).
* **Filter/Kernel (K):** A **3x3** grid of weights.
* **Stride:** 1.
* **Padding:** 0.

### Step 1: The Convolution Scan
We slide the **3x3 Filter** over the **5x5 Image**.
* **Output Size Math:** `(5 - 3)/1 + 1` = **3x3 Output**.

**Snapshot 1: Top-Left Patch**
The filter overlays the top-left 3x3 region. We perform the Dot Product.

```text
       INPUT IMAGE (5x5)              FILTER (3x3)
    ┌───┬───┬───┬───┬───┐           ┌───┬───┬───┐
    │ A │ B │ C │ . │ . │           │ w │ x │ w │
    ├───┼───┼───┼───┼───┤           ├───┼───┼───┤
    │ F │ G │ H │ . │ . │     x     │ x │ w │ x │
    ├───┼───┼───┼───┼───┤           ├───┼───┼───┤
    │ K │ L │ M │ . │ . │           │ w │ x │ w │
    ├───┼───┼───┼───┼───┤           └───┴───┴───┘
    │ . │ . │ . │ . │ . │
    └───┴───┴───┴───┴───┘
    
    Calculation: (A*w) + (B*x) + (C*w) + ... + (M*w)
    Result ──▶ Output Map pixel [0,0]
```

**Snapshot 2: Slide Right**
We move 1 pixel to the right.

```text
       INPUT IMAGE (5x5)              FILTER (3x3)
    ┌───┬───┬───┬───┬───┐           ┌───┬───┬───┐
    │ . │ B │ C │ D │ . │           │ w │ x │ w │
    ├───┼───┼───┼───┼───┤           ├───┼───┼───┤
    │ . │ G │ H │ I │ . │     x     │ x │ w │ x │
    ├───┼───┼───┼───┼───┤           ├───┼───┼───┤
    │ . │ L │ M │ N │ . │           │ w │ x │ w │
    ├───┼───┼───┼───┼───┤           └───┴───┴───┘
    │ . │ . │ . │ . │ . │
    └───┴───┴───┴───┴───┘

    Calculation: Sum of products...
    Result ──▶ Output Map pixel [0,1]
```

*(This process repeats 9 times total until we cover the whole image)*


### Step 2: The Feature Map & Activation
Let's assume the convolution is finished. We now have a **3x3 Feature Map**.
We immediately apply **ReLU** (`max(0, x)`) to remove negative values.

```text
    Raw Feature Map (Z)         ReLU Activation (A)
    ┌─────┬─────┬─────┐         ┌─────┬─────┬─────┐
    │  10 │ -40 │  20 │         │  10 │   0 │  20 │
    ├─────┼─────┼─────┤   ──▶  ├─────┼─────┼─────┤
    │ -10 │  50 │  30 │         │   0 │  50 │  30 │
    ├─────┼─────┼─────┤         ├─────┼─────┼─────┤
    │   5 │  15 │ -90 │         │   5 │  15 │   0 │
    └─────┴─────┴─────┘         └─────┴─────┴─────┘
```

### Step 3: Pooling (Downsampling)
We apply **Max Pooling**. Let's use a 2x2 window with Stride 1 for demonstration (or usually Stride 2 to shrink it).
*For simplicity here, let's just take the Global Max of the quadrants or shrink the 3x3 to a 2x2.*

**Resulting Pooled Map:**
```text
    ┌─────┬─────┐
    │  50 │  30 │
    ├─────┼─────┤
    │  15 │   5 │
    └─────┴─────┘
```

### Step 4: Flatten & Predict
We turn the grid into a vector and feed it to a standard Linear Layer.

1.  **Flatten:** `[50, 30, 15, 5]`
2.  **Dense Weights:** Multiply by classification weights.
3.  **Score:** `0.8` (e.g., probability of "Cat").

## The Backward Pass (Learning)

The network predicted "Cat" (0.8). But the image was actually a "Dog".
**We have an Error.** Now we must train the Filter to do a better job next time.

### 1. Calculate the Loss
We compare Prediction vs Target using a Loss Function (e.g., Cross Entropy or MSE).

> $$\text{Loss} = (\text{Prediction} - \text{Target})^2$$

### 2. Backpropagation (Passing the Blame)
We send the error signal backwards through the network:
1.  **Dense Layer:** "Hey, I weighted the 'ears' feature too high." (Update Dense Weights).
2.  **Flatten/Pool:** Un-pool the error back to the Feature Map locations.
3.  **ReLU:** If the pixel was negative (killed), it gets 0 blame. If positive, it passes the blame through.

### 3. Updating the Filter (The Critical Step)
This is where CNNs are special. We need to update the **3x3 Filter Weights**.

**The Challenge:**
The weight `W[0,0]` (top-left of filter) was used **9 times** during the scan. It contributed to the error in the top-left, top-right, center, etc.

**The Solution (Accumulate Gradients):**
To find the gradient for a specific filter weight, we sum up the blame from **every position** it touched.

> $$\frac{\partial L}{\partial w} = \sum (\text{Input Patch} \cdot \text{Output Error})$$


### Visualizing the Gradient Accumulation for $w_{0,0}$

Imagine the **Filter** is sliding over the **Input** again. We are watching **only the Top-Left corner** of the filter (marked as `*`).

**The Goal:** Find $\nabla w_{0,0}$ (How much to change the top-left weight).

```text
    THE FILTER (3x3)
    ┌───┬───┬───┐
    │ * │ . │ . │   <── We are tracking ONLY this weight
    ├───┼───┼───┤
    │ . │ . │ . │
    └───┴───┴───┘
```

**Replaying the Scan:**

**1. Position 1 (Top-Left)**
The filter sits on the start of the image. Our weight `*` is on top of pixel `A`.
* **The Error here:** $E_{1}$ (From Output[0,0])
* **Contribution:** $A \cdot E_{1}$

```text
    INPUT IMAGE                  OUTPUT ERROR MAP
    ┌───┬───┬───┬──┐             ┌────┬────┬────┐
    │ A │ . │ . │ .│             │ E1 │ .  │ .  │
    ├───┼───┼───┼──┤             ├────┼────┼────┤
    │ . │ . │ . │ .│             │ .  │ .  │ .  │
      ▲                            ▲
      └── "I touched A"            └── "And we made Error E1"
```

**2. Position 2 (Slide Right)**
The filter slides right. Now our weight `*` is on top of pixel `B`.
* **The Error here:** $E_{2}$ (From Output[0,1])
* **Contribution:** $B \cdot E_{2}$

```text
    INPUT IMAGE                  OUTPUT ERROR MAP
    ┌───┬───┬───┬──┐             ┌────┬────┬────┐
    │ . │ B │ . │ .│             │ .  │ E2 │ .  │
    ├───┼───┼───┼──┤             ├────┼────┼────┤
    │ . │ . │ . │ .│             │ .  │ .  │ .  │
          ▲                             ▲
          └── "I touched B"             └── "And we made Error E2"
```

**3. Position 3 (Slide Right)**
The filter slides again. Weight `*` is on top of pixel `C`.
* **The Error here:** $E_{3}$ (From Output[0,2])
* **Contribution:** $C \cdot E_{3}$

```text
    INPUT IMAGE                  OUTPUT ERROR MAP
    ┌───┬───┬───┬──┐             ┌────┬────┬────┐
    │ . │ . │ C │ .│             │ .  │ .  │ E3 │
    ├───┼───┼───┼──┤             ├────┼────┼────┤
    │ . │ . │ . │ .│             │ .  │ .  │ .  │
              ▲                              ▲
              └── "I touched C"              └── "And we made Error E3"
```

**The Final Calculation**
We sum up every interaction to find the **Total Gradient**.

$$\nabla w_{0,0} = (A \cdot E_1) + (B \cdot E_2) + (C \cdot E_3) + \dots$$

* **Logic:** "If pixel `A` was bright and we had a huge error `E1`, then weight `w` must change a lot. If pixel `B` was black (0), then `w` didn't contribute anything there, so ignore `E2`."

**The Update Formula (Gradient Descent):**
Once we have the total gradient for every pixel in the 3x3 filter, we update them:

> $$W_{new} = W_{old} - (\text{Learning Rate} \cdot \text{Total Gradient})$$

* **Result:** The filter slowly changes from random noise into a structured pattern (like an edge detector) that minimizes the error.

## Why CNNs Learn Better (The "Shared Weights" Magic)

In a standard Neural Network (MLP), if you trained it to recognize a cat in the **top-left** corner, it would learn specific weights for those top-left pixels. If you then moved the cat to the **bottom-right**, the MLP would fail—it has to learn "cat in bottom-right" as a totally new concept.

**CNNs solve this via Parameter Sharing:**
Because the **same Filter ($K$)** slides over the whole image:
1.  **Efficiency:** We learn far fewer parameters (just the small filter weights).
2.  **Translation Invariance:** Once the filter learns what an "Eye" looks like, it can find an eye *anywhere* in the image (top, bottom, left, or right).

</details>
