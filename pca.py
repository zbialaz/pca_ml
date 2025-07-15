import numpy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# Generate a dataset with synthetic data with 3 variables
X, _ = make_classification(n_samples=200, n_features=3, n_informative=3, n_redundant=0, random_state=42)

# Apply PCA to reduce the dataset to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the before and after PCA
fig = plt.figure(figsize=(12, 5))

# Original Data in 3D
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', alpha=0.6)
ax1.set_title('Original Data in 3D')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('X3')

# Data after PCA in 2D
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='red', alpha=0.6)
ax2.set_title('Data after PCA in 2D')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

plt.tight_layout()
plt.show()