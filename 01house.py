# Requirements: pandas, scikit-learn, matplotlib, numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Example data (replace with your dataset)
data = {
    "area": [600, 800, 1000, 1200, 1500, 1800, 2000],
    "price": [150000, 190000, 230000, 270000, 330000, 380000, 420000]
}
df = pd.DataFrame(data)

# Features / target
X = df[["area"]]          # 2D array
y = df["price"]           # 1D array

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients
b0 = model.intercept_
b1 = model.coef_[0]
print(f"Intercept (b0): {b0:.2f}")
print(f"Slope (b1): {b1:.2f}  (price per unit area)")

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R^2: {r2:.3f}")

# Example prediction
new_area = np.array([[1400]])
pred_price = model.predict(new_area)[0]
print(f"Predicted price for area={new_area[0,0]}: {pred_price:.2f}")

# Optional: plot
plt.scatter(X, y, label="Data")
plt.plot(X, model.predict(X), color="red", label="Linear fit")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()
