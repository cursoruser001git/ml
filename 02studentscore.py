# Requirements: pandas, scikit-learn, matplotlib, numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Example data (replace with your dataset)
data = {
    "study_hours": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7],
    "marks":       [50,55,58,62,65,68,70,75,80,85]
}
df = pd.DataFrame(data)

# Features / target
X = df[["study_hours"]]
y = df["marks"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients
b0 = model.intercept_
b1 = model.coef_[0]
print(f"Intercept (b0): {b0:.3f}")
print(f"Slope (b1): {b1:.3f}  (marks per study hour)")

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.3f}")
print(f"R^2: {r2:.3f}")

# Example prediction
new_hours = np.array([[4.2]])
pred_marks = model.predict(new_hours)[0]
print(f"Predicted marks for study_hours={new_hours[0,0]}: {pred_marks:.2f}")

# Plot
plt.scatter(X, y, label="Data")
plt.plot(X, model.predict(X), color="red", label="Linear fit")
plt.xlabel("Study hours")
plt.ylabel("Marks")
plt.legend()
plt.show()
