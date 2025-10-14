import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.distutils.misc_util import colour_text
from sklearn.linear_model import LinearRegression, Ridge
np.random.seed(42)
n = 100
df = pd.DataFrame({
    "study_hours": np.random.randint(1, 10, n),
    "attendance": np.random.randint(60, 100, n),
    "previous_scores": np.random.randint(40, 100, n),
    "participation": np.random.randint(1, 10, n)})

df["final_score"] = (
    df["study_hours"] * 5 +
    df["attendance"] * 0.5 +
    df["previous_scores"] * 0.3 +
    df["participation"] * 2 +
    np.random.normal(0, 10, n))

X = df[["study_hours", "attendance", "previous_scores", "participation"]]
y = df["final_score"]

model = LinearRegression()
model.fit(X, y)

print("Linear Regression Intercept:", model.intercept_)
print("Linear Regression Coefficients:", model.coef_)

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

print("Ridge Regression Intercept:", ridge.intercept_)
print("Ridge Regression Coefficients:", ridge.coef_)

new_student = pd.DataFrame([[8, 90, 85, 7]], columns=X.columns)

predicted_linear = model.predict(new_student)
predicted_ridge = ridge.predict(new_student)

print("Predicted score from Linear Regression:", predicted_linear[0])
print("Predicted score from Ridge Regression:", predicted_ridge[0])

plt.scatter(y, model.predict(X), label="Linear", alpha=0.6)
plt.scatter(y, ridge.predict(X), label="Ridge", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Student Performance Prediction")
plt.legend()
plt.show()
