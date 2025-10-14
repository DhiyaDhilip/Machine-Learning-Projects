# 📦 Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

# 🎲 Set random seed for reproducibility
np.random.seed(42)
n = 100

# 📊 Generate synthetic dataset
df = pd.DataFrame({
    "experience": np.random.randint(0, 20, n),         # Years of experience
    "education": np.random.randint(1, 4, n),            # 1: High School, 2: Bachelor's, 3: Master's
    "job_role": np.random.randint(1, 6, n),             # Encoded job roles
    "location": np.random.randint(1, 4, n)              # Encoded locations
})

# 🎯 Target: Salary (in thousands)
df["salary"] = (
    df["experience"] * 5 +
    df["education"] * 10 +
    df["job_role"] * 8 +
    df["location"] * 3 +
    np.random.normal(0, 10, n)
)

# 🧹 Select features and target
X = df[["experience", "education", "job_role", "location"]]
y = df["salary"]

# 📈 Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# 🔍 Print Linear Regression parameters
print("Linear Regression Intercept:", model.intercept_)
print("Linear Regression Coefficients:", model.coef_)

# 🛡️ Train Ridge Regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# 🔍 Print Ridge Regression parameters
print("Ridge Regression Intercept:", ridge.intercept_)
print("Ridge Regression Coefficients:", ridge.coef_)

# 👩‍💼 New employee input
new_employee = pd.DataFrame([[5, 2, 3, 2]], columns=X.columns)

# 🔮 Predict salary
predicted_linear = model.predict(new_employee)
predicted_ridge = ridge.predict(new_employee)

# 📢 Display predictions
print("Predicted salary from Linear Regression:", predicted_linear[0])
print("Predicted salary from Ridge Regression:", predicted_ridge[0])

# 📊 Visualization
# 📊 Bar chart: Actual vs Predicted Salary
y_pred_linear = model.predict(X)
y_pred_ridge = ridge.predict(X)

# Select first 20 samples for clarity
indices = range(20)
actual = y.iloc[indices].values
pred_linear = y_pred_linear[indices]
pred_ridge = y_pred_ridge[indices]

# Plot
bar_width = 0.3
x = np.arange(len(indices))

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width, actual, width=bar_width, label="Actual", color="gray")
plt.bar(x, pred_linear, width=bar_width, label="Linear", color="skyblue")
plt.bar(x + bar_width, pred_ridge, width=bar_width, label="Ridge", color="orange")

plt.xlabel("Employee Index")
plt.ylabel("Salary")
plt.title("Actual vs Predicted Salary (Bar Chart)")
plt.xticks(x, [f"E{i}" for i in indices])
plt.legend()
plt.tight_layout()
plt.show()
