# Greedy Two-Layer Neural Network Approximation

This repository contains a PyTorch implementation and empirical evaluation of the **iterative greedy neural network approximation algorithm** introduced by **Lee K. Jones (1992)** in the seminal paper:
> *"[A Simple Lemma on Greedy Approximation in Hilbert Space and Convergence Rates for Projection Pursuit Regression and Neural Network Approximation](https://ieeexplore.ieee.org/document/142036)"*

The goal is to construct a shallow two-layer neural network with $n$ hidden units to approximate a target function $f(x)$ defined on $[-1, 1]$:

$$y(x) = \sum_{i=1}^n c_i \sigma(w_i x + b_i)$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the standard sigmoid activation function. The Jones greedy algorithm guarantees that under suitable conditions, the $L^2$ approximation error decays at a rate of $\mathcal{O}(n^{-1/2})$.

---

## 📐 Algorithm Formulation

Let $h_0(x) = 0$ denote the initial approximation. The greedy iterations are computed as follows for step $k \ge 0$:

1. **Find the parameter pair $(w_k, b_k) \in \mathbb{R}^2$** that satisfies the descent inequality.
2. **Determine the optimal scaling factor $\alpha_k \in [0, 1]$** to update the network:

$$h_{k+1}(x) = (1-\alpha_k) h_k(x) + \alpha_k \sigma(w_k x + b_k)$$

### Key Implementation Details

#### 1. Optimal Scaling Factor ($\alpha_k$) Calculation
Instead of using iterative numerical solvers to determine the step size $\alpha_k$, the optimal unconstrained scaling factor $\alpha^*$ is derived analytically. Let:
* $A(x) = f(x) - h_k(x)$ (the current approximation error)
* $B(x) = \sigma_k(x) - h_k(x)$ (the update direction)

We minimize the quadratic objective $\| A - \alpha B \|^2_{L^2(X)}$. Setting the derivative with respect to $\alpha$ to zero yields:

$$\alpha^* = \frac{\langle A, B \rangle}{\|B\|^2} = \frac{\langle f - h_k, \sigma_k - h_k \rangle}{\|\sigma_k - h_k\|^2}$$

Since the update must represent a convex combination, the scaling factor $\alpha_k$ must lie within $[0, 1]$. We project $\alpha^*$ onto the constraint set:

$$\alpha_k = \max\left(0, \min\left(1, \alpha^*\right)\right)$$

#### 2. Sigmoid Selection via Random Sampling
Finding the global minimizer $(w_k, b_k)$ is non-convex and computationally challenging. However, the convergence proof only requires finding a sigmoid unit $\sigma_k(x) = \sigma(w_k x + b_k)$ that satisfies the **Jones descent inequality**:

$$\langle f(x) - h_k(x), f(x) - \sigma(w_k x + b) \rangle < 0$$

In our implementation, this is achieved by randomly sampling weights and biases $w, b \sim \mathcal{U}(-10, 10)$ up to 5,000 times per iteration. The first candidate to satisfy the inequality is chosen immediately. If no candidate satisfies it within 5,000 trials, the algorithm falls back to the candidate that got closest (i.e., minimized the inner product).

---

## 📂 Codebase Structure

* **`greedy_two_layer_nn.py`**: Contains the core logic and network architecture.
  * `TwoLayerNet`: PyTorch class representing a shallow two-layer network with a high hidden dimension (1000 units), used to generate random target functions in the convex hull of sigmoids.
  * `loss(y_output, h_output)`: Implements the mean squared error (MSE) loss.
  * `get_alpha(y_output, h_output, sigma_output)`: Computes the optimal analytical scaling factor, clipped to $[0, 1]$.
  * `minimize_algorithm(x_inputs, y_output, n)`: Runs the greedy approximation algorithm for $n$ terms.
* **`run_experiments.py`**: Executes the experiment suite, logging the results and saving the error decay plot.
* **`problem.md`**: Formal definition of the algorithm, constraints, and experimental guidelines.
* **`proof.md`**: Mathematical exercises detailing the step-by-step convergence proof.

---

## 🚀 How to Run the Experiments

### 1. Prerequisites
Ensure you have Python 3 and the required libraries installed:
```bash
pip install torch matplotlib numpy
```

### 2. Execution
Run the experiment script to execute all 20 independent trials and regenerate the error decay visualization:
```bash
python3 run_experiments.py
```
This will print the mean MSE, mean $L^2$ error (RMSE), and standard deviations to the terminal, and save the updated plot as `error_decay_plot.png` in the root directory.
