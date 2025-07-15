# PCA Machine Learning - Principal Component Analysis

This project demonstrates the application of **PCA (Principal Component Analysis)** for dimensionality reduction on synthetic data, visualizing the transformation from 3D to 2D data.

## 📊 About the Project

PCA is a fundamental machine learning technique used to:
- Reduce data dimensionality
- Preserve the maximum amount of information possible
- Facilitate visualization of complex data
- Improve ML algorithm performance

## 🚀 Features

- **Synthetic data generation**: Creating 3-dimensional dataset using `make_classification`
- **PCA application**: Reducing from 3D to 2D while maintaining the most important information
- **Comparative visualization**: Side-by-side plots showing original and transformed data

## 📋 Prerequisites

```bash
pip install numpy matplotlib scikit-learn
```

## 🔧 How to Use

1. Clone the repository:
```bash
git clone https://github.com/zbialaz/pca_machine_learning.git
cd pca_machine_learning
```

2. Run the script:
```bash
python pca.py
```

## 📈 Results

The script generates two visualizations:

### Original Data (3D)
- 3-dimensional scatter plot
- 200 data points
- 3 informative features

### Data after PCA (2D)
- 2-dimensional scatter plot
- Same 200 transformed points
- 2 principal components (PC1 and PC2)

## 🛠️ Dataset Parameters

- `n_samples=200`: Number of generated samples
- `n_features=3`: Number of features (dimensions)
- `n_informative=3`: All features are informative
- `n_redundant=0`: No redundant features
- `random_state=42`: Seed for reproducibility

## 📊 Code Structure

```python
# 1. Import libraries
import numpy, matplotlib.pyplot, sklearn

# 2. Generate synthetic data
X, _ = make_classification(...)

# 3. Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 4. Visualize results
# Subplot 1: Original 3D data
# Subplot 2: Transformed 2D data
```

## 📚 Concepts Learned

- **PCA (Principal Component Analysis)**: Dimensionality reduction technique
- **Principal Components**: Directions of maximum variance in the data
- **3D vs 2D Visualization**: Comparison between dimensionalities
- **Scikit-learn**: Using the library for machine learning

## 📧 Contact

- **GitHub**: [@zbialaz](https://github.com/zbialaz)
- **Project**: [pca_machine_learning](https://github.com/zbialaz/pca_machine_learning)

---
