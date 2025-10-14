import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Step 1: Load California housing data
X, y = fetch_california_housing(return_X_y=True)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("First 5 rows of X:\n", X[:5])
print("First 5 values of y:\n", y[:5])

# Step 2: Scale the features
X = StandardScaler().fit_transform(X)

# Step 3: Define alpha values
alphas = np.linspace(0.001, 1000, 100)
coefs = []

# Step 4: Fit Ridge model for each alpha and store coefficients
for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X, y)
    coefs.append(model.coef_)

# Step 5: Plot coefficients vs alpha
plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Ridge Coefficients vs Alpha (California Housing)')
plt.tight_layout()
plt.grid(True)
plt.show()