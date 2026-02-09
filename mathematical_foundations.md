# Linear Algebra for Machine Learning

## 1. Introduction
Linear algebra forms the mathematical foundation of machine learning. Most ML algorithms can be expressed as matrix operations.

## 2. Vectors

### Definition
A vector $\mathbf{v} \in \mathbb{R}^n$ is an ordered list of $n$ real numbers:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
$$

### Operations
- **Addition**: $\mathbf{u} + \mathbf{v} = [u_1 + v_1, u_2 + v_2, \dots, u_n + v_n]^T$
- **Scalar Multiplication**: $c\mathbf{v} = [cv_1, cv_2, \dots, cv_n]^T$
- **Dot Product**: $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i$

### Geometric Interpretation
Vectors can represent:
- Points in space
- Directions and magnitudes
- Features in ML datasets

## 3. Matrices

### Definition
A matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a rectangular array:

$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

### Types of Matrices
1. **Square Matrix**: $m = n$
2. **Diagonal Matrix**: $a_{ij} = 0$ for $i \neq j$
3. **Identity Matrix**: Diagonal matrix with ones
4. **Symmetric Matrix**: $\mathbf{A} = \mathbf{A}^T$

## 4. Matrix Operations

### Multiplication
For $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$:

$$
\mathbf{C} = \mathbf{A}\mathbf{B} \quad \text{where} \quad c_{ij} = \sum_{k=1}^n a_{ik} b_{kj}
$$

### Transpose
The transpose $\mathbf{A}^T$ flips rows and columns:

$$
(\mathbf{A}^T)_{ij} = a_{ji}
$$

### Inverse
For a square matrix $\mathbf{A}$, its inverse $\mathbf{A}^{-1}$ satisfies:

$$
\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}
$$

## 5. Eigenvalues and Eigenvectors

### Definition
For a square matrix $\mathbf{A}$, a non-zero vector $\mathbf{v}$ is an eigenvector and $\lambda$ is an eigenvalue if:

$$
\mathbf{A}\mathbf{v} = \lambda\mathbf{v}
$$

### Properties
- Characteristic equation: $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$
- Trace: $\text{tr}(\mathbf{A}) = \sum_i \lambda_i$
- Determinant: $\det(\mathbf{A}) = \prod_i \lambda_i$

## 6. Singular Value Decomposition (SVD)

### Theorem
Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T
$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: Left singular vectors (orthogonal)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: Diagonal matrix of singular values
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: Right singular vectors (orthogonal)

### Applications in ML
1. **PCA**: Dimensionality reduction
2. **Matrix Completion**: Recommender systems
3. **Latent Semantic Analysis**: NLP

## 7. Applications in Machine Learning

### Linear Regression
The normal equation:

$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

### Principal Component Analysis (PCA)
Find eigenvectors of covariance matrix:

$$
\mathbf{C} = \frac{1}{n}\mathbf{X}^T\mathbf{X}
$$

### Neural Networks
Forward propagation:

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
$$

## 8. Python Implementation

```python
import numpy as np

# Basic operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, S, Vt = np.linalg.svd(A)

# Solving linear systems
# Ax = b
x = np.linalg.solve(A, b)
```
