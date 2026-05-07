# Requirements: scikit-learn, pandas, numpy, matplotlib (optional)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Example dataset (replace with your data or upload)
data = {
    "years_experience": [1,2,3,4,5,6,7,8,9,10],
    "salary":           [45000,50000,60000,65000,70000,76000,82000,90000,98000,105000]
}
df = pd.DataFrame(data)

# Features / target
X = df[["years_experience"]]
y = df["salary"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients
b0 = model.intercept_
b1 = model.coef_[0]
print(f"Intercept (b0): {b0:.2f}")
print(f"Slope (b1): {b1:.2f}  (salary increase per year)")

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R^2: {r2:.3f}")

# Example prediction
new_years = np.array([[4.5]])
pred_salary = model.predict(new_years)[0]
print(f"Predicted salary for {new_years[0,0]} years experience: ${pred_salary:,.2f}")

# Optional plot
plt.scatter(X, y, label="Data")
plt.plot(X, model.predict(X), color="red", label="Linear fit")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
