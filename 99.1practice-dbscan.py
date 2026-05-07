import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# 1. Create the Data (Unsupervised: Interlocking moons with some random noise)
X_data, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
df = pd.DataFrame(X_data, columns=['Feature_1', 'Feature_2'])

# 2. Initialize the DBSCAN Model
# eps: The maximum distance between two points to be considered neighbors
# min_samples: The number of neighbors required to form a dense "Core"
dbscan = DBSCAN(eps=0.2, min_samples=5)

# 3. Train the Model and Get Predictions
# (DBSCAN doesn't have a separate .predict() for new data, we use fit_predict)
labels = dbscan.fit_predict(df)

# ==========================================
# 4. Identify Core, Border, and Noise Points
# ==========================================

# A. Find the Core Points (sklearn gives us their exact index numbers)
core_mask = np.zeros_like(labels, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True

# B. Find the Noise Points (sklearn always labels noise as -1)
noise_mask = (labels == -1)

# C. Find the Border Points (Everything that isn't Core and isn't Noise)
border_mask = ~(core_mask | noise_mask)

# Print the counts for the exam!
print(f"Total Points:  {len(df)}")
print(f"Core Points:   {np.sum(core_mask)}")
print(f"Border Points: {np.sum(border_mask)}")
print(f"Noise Points:  {np.sum(noise_mask)}")

# 5. Plot the Results
plt.scatter(df['Feature_1'][core_mask], df['Feature_2'][core_mask], 
            c='blue', label='Core', s=50)
plt.scatter(df['Feature_1'][border_mask], df['Feature_2'][border_mask], 
            c='green', label='Border', s=20)
plt.scatter(df['Feature_1'][noise_mask], df['Feature_2'][noise_mask], 
            c='red', label='Noise (Outliers)', marker='x')

plt.title('DBSCAN Clustering')
plt.legend()
plt.show()