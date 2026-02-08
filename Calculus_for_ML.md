
## 💻 **Step 3: Create Practical Implementation**

Create `practical/beginner/01_linear_regression/linear_regression.md`:

```markdown
# Linear Regression: Theory and Implementation

## 1. Mathematical Foundation

### Problem Formulation
Given dataset $\{(x_i, y_i)\}_{i=1}^n$, find parameters $w$ and $b$ that minimize:

$$
J(w, b) = \frac{1}{2n} \sum_{i=1}^n (y_i - (wx_i + b))^2
$$

### Normal Equation Solution
For the matrix formulation $\mathbf{y} = \mathbf{X}\mathbf{w}$:

$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

### Gradient Descent Update Rules
- **Batch Gradient Descent**:
  $$ w := w - \alpha \frac{\partial J}{\partial w} $$
  $$ b := b - \alpha \frac{\partial J}{\partial b} $$

- **Stochastic Gradient Descent**:
  Update using single training example

## 2. Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iters):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Compute loss
            loss = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.loss_history.append(loss)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)
```

# 4. Mathematical Derivation
# Cost Function Derivation
The mean squared error:

J
(
θ
)
=
1
2
m
∑
i
=
1
m
(
h
θ
(
x
(
i
)
)
−
y
(
i
)
)
2
J(θ)= 
2m
1
​
  
i=1
∑
m
​
 (h 
θ
​
 (x 
(i)
 )−y 
(i)
 ) 
2
 
where $h_\theta(x) = \theta^T x$.

# Gradient Calculation
# Partial derivatives:

∂
J
∂
θ
j
=
1
m
∑
i
=
1
m
(
h
θ
(
x
(
i
)
)
−
y
(
i
)
)
x
j
(
i
)
∂θ 
j
​
 
∂J
​
 = 
m
1
​
  
i=1
∑
m
​
 (h 
θ
​
 (x 
(i)
 )−y 
(i)
 )x 
j
(i)
​
 
In matrix form:

∇
θ
J
(
θ
)
=
1
m
X
T
(
X
θ
−
y
)
∇ 
θ
​
 J(θ)= 
m
1
​
 X 
T
 (Xθ−y)
# Normal Equation Derivation
Set gradient to zero:

∇
θ
J
(
θ
)
=
0
∇ 
θ
​
 J(θ)=0
X
T
(
X
θ
−
y
)
=
0
X 
T
 (Xθ−y)=0
X
T
X
θ
=
X
T
y
X 
T
 Xθ=X 
T
 y
θ
=
(
X
T
X
)
−
1
X
T
y
θ=(X 
T
 X) 
−1
 X 
T
 y
# 5. Advanced Topics
# Regularization
# Ridge Regression (L2 regularization):

J
(
θ
)
=
1
2
m
[
∑
i
=
1
m
(
h
θ
(
x
(
i
)
)
−
y
(
i
)
)
2
+
λ
∑
j
=
1
n
θ
j
2
]
J(θ)= 
2m
1
​
 [ 
i=1
∑
m
​
 (h 
θ
​
 (x 
(i)
 )−y 
(i)
 ) 
2
 +λ 
j=1
∑
n
​
 θ 
j
2
​
 ]
# Lasso Regression (L1 regularization):

J
(
θ
)
=
1
2
m
[
∑
i
=
1
m
(
h
θ
(
x
(
i
)
)
−
y
(
i
)
)
2
+
λ
∑
j
=
1
n
∣
θ
j
∣
]
J(θ)= 
2m
1
​
 [ 
i=1
∑
m
​
 (h 
θ
​
 (x 
(i)
 )−y 
(i)
 ) 
2
 +λ 
j=1
∑
n
​
 ∣θ 
j
​
 ∣]
# Polynomial Regression
Extend to polynomial features:

h
θ
(
x
)
=
θ
0
+
θ
1
x
+
θ
2
x
2
+
⋯
+
θ
n
x
n
h 
θ
​
 (x)=θ 
0
​
 +θ 
1
​
 x+θ 
2
​
 x 
2
 +⋯+θ 
n
​
 x 
n
 
# 6. Assumptions and Diagnostics
# Gauss-Markov Assumptions
Linearity

Independence

Homoscedasticity

No perfect multicollinearity

Normally distributed errors

