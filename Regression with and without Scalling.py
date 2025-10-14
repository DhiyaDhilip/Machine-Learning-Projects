import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Step 1: Create a simple dataset
data = pd.DataFrame({
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Sleep_Hours': [9, 8, 7, 6, 6, 5, 4, 3],
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1]
})

X = data[['Study_Hours', 'Sleep_Hours']]
y = data['Pass']

# Step 2: Logistic Regression WITHOUT scaling
model_no_scale = LogisticRegression()
model_no_scale.fit(X, y)

# Step 3: Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Logistic Regression WITH scaling
model_scaled = LogisticRegression()
model_scaled.fit(X_scaled, y)

# Step 5: Compare coefficients
comparison = pd.DataFrame({
    'Feature': ['Study_Hours', 'Sleep_Hours'],
    'Coeff (No Scaling)': model_no_scale.coef_[0],
    'Coeff (Scaled)': model_scaled.coef_[0]
})

print(comparison)