# Custom Neural Network from Scratch (NumPy)

This repository contains a clean, from-scratch implementation of an Artificial Neural Network (ANN) using Python and NumPy. The project is designed to explicitly showcase the mathematical mechanics of deep learning—specifically forward propagation, weight initialization strategies, and backpropagation—without relying on heavy deep-learning frameworks like PyTorch or TensorFlow.

## Repository Structure

* **`src/NeuralNetworkSimple.py`**: The foundational implementation exposing the raw math, loops, and matrix transformations for clear understanding.
* **`src/NeuralNetwork.py`**: A fully optimized, parallelized version leveraging NumPy's advanced vectorization capabilities for speed.
* **`src/utilz.py`**: Custom data-preprocessing utilities including stable stratified splitting.
* **`src/temp.py`**: Sandboxed environment testing the network on the MNIST dataset.

---

## Architecture & Mathematical Foundations

### 1. Layer and Weight Notation
The network treats weights as structured transformation matrices between layers. For any layer index $l$:

* The activation layer is represented as a vector $a^{(l)}$.
* The weight matrix $W^{(l)}$ connects layer $l$ to layer $l+1$. 
* To keep row-major operations natural, the matrix dimensions are defined as:

$$\text{Shape of } W^{(l)} = (n_{l+1}, n_l)$$

Where $n_{l+1}$ is the number of neurons in the destination layer, and $n_l$ is the number of neurons in the source layer. An individual weight element $W_{jk}^{(l)}$ maps the $k$-th neuron of layer $l$ to the $j$-th neuron of layer $l+1$.

---

### 2. Weight Initialization Strategies
To prevent gradients from exploding or vanishing during backpropagation, weights are automatically scaled down based on the incoming layer's size $n_{in}$ and the choice of activation function:

* **He Initialization** (Optimized for **ReLU**):
    Weights are sampled from a standard normal distribution and scaled by:
    $$\sigma = \sqrt{\frac{2.0}{n_{in}}}$$

* **Xavier Initialization** (Optimized for **Sigmoid**):
    Weights are sampled from a standard normal distribution and scaled by:
    $$\sigma = \sqrt{\frac{1.0}{n_{in}}}$$

---

### 3. Forward Propagation
For each layer, the net input vector $z^{(l)}$ is computed by mapping the previous layer's activations through the weight matrix and adding the bias vector $b^{(l)}$:

$$z^{(l)} = W^{(l-1)} a^{(l-1)} + b^{(l-1)}$$

The activation vector for the layer is then computed via the chosen non-linear activation function $f$:

$$a^{(l)} = f(z^{(l)})$$

The network natively handles a dedicated activation function for the hidden layers (e.g., ReLU) and a distinct one for the output layer (e.g., Sigmoid for probability distribution).

---

### 4. Backpropagation & Gradient Calculus
The network minimizes the Total Squared Error (Cost Function $C$) across the network predictions:

$$C = \sum (a^{(L)} - y)^2$$

Where $a^{(L)}$ is the final layer output prediction and $y$ is the one-hot encoded ground truth.

#### Output Layer Error ($\delta^{(L)}$)
The error gradient with respect to the output activation space is computed first:

$$\delta^{(L)} = \frac{\partial C}{\partial z^{(L)}} = 2(a^{(L)} - y) \odot f_L'(z^{(L)})$$

Where $\odot$ represents the element-wise Hadamard product, and $f_L'$ is the derivative of the final layer's activation function.

#### Error Backpropagation (Hidden Layers)
The error vector $\delta$ is recursively passed backward through the network layers from $l+1$ to $l$:

$$\delta^{(l)} = \left( (W^{(l)})^T \delta^{(l+1)} \right) \odot f'(z^{(l)})$$

#### Gradient Accumulation
Using these errors, the exact partial derivatives required to tune the biases and weights are derived via outer products ($\otimes$):

$$\frac{\partial C}{\partial b^{(l)}} = \delta^{(l+1)}$$

$$\frac{\partial C}{\partial W^{(l)}} = \delta^{(l+1)} \otimes a^{(l)}$$

### A Deeper Look: The Outer Product Equivalence

In the implementation (`NeuralNetworkSimple.py`), the weight gradient is computed using `np.outer(delta, self.layers[i-1])`. 
Mathematically, this elegant trick avoids nested loops by broadcasting the error vector across the activation states of the previous layer.

For any weight $W_{jk}^{(l)}$ connecting neuron $k$ in layer $l$ to neuron $j$ in layer $l+1$, the individual gradient is defined as:

$$\frac{\partial C}{\partial W_{jk}^{(l)}} = \delta_j^{(l+1)} \cdot a_k^{(l)}$$

If we look at this from a matrix operations perspective, calculating this for every single $j$ (rows) and $k$ (columns) is structurally equivalent to performing a matrix multiplication between a column vector and a row vector:

$$\frac{\partial C}{\partial W^{(l)}} = \delta^{(l+1)} (a^{(l)})^T$$

> **Implementation Intuition:** > This is exactly what the code does. Instead of manually constructing a matching transformation matrix `temp` where each element holds the corresponding activation $a_k^{(l)}$ 
> and performing a traditional matrix dot product (`delta @ temp`), `np.outer()` takes the vector $\delta^{(l+1)}$ and multiplies it across the transposed activation vector $(a^{(l)})^T$ in a single, highly optimized step. 
> This instantly yields the complete gradient matrix matching the shape of $W^{(l)}$.

#### Optimization Update
Using Online Stochastic Gradient Descent (SGD), the network parameters update over a learning rate $\alpha$:

$$W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial C}{\partial W^{(l)}}$$

$$b^{(l)} \leftarrow b^{(l)} - \alpha \frac{\partial C}{\partial b^{(l)}}$$

---

## Advanced Utilities

### Stable Stratified Splitting
Located inside `src/utilz.py`, the `train_test_split` function guarantees deterministic and perfectly distributed splits across data updates without random state drift. It creates a stable unique key combining the row index and the stratification column target, hashing it via Cyclic Redundancy Check (`crc32`):

$$\text{Condition for Test Set} = \left( \text{crc32}(\text{Target} \parallel \text{Index}) \ \& \ \text{0xffffffff} \right) < \text{ratio} \times 2^{32}$$

This ensures that even if new data rows are appended to the dataset, your training and testing sets remain structurally uncorrupted and reproducible.

---

## Quick Start

### Installation
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```
### Training the Model

To run the placeholder training sequence using the MNIST dataset:

* Place your mnist_train.csv dataset into a src/data/ directory.
* Run the pipeline script:
```bash
python -m src.temp
```