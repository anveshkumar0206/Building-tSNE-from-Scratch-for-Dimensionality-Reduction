# Building-tSNE-from-Scratch-for-Dimensionality-Reduction

This project implements t-SNE (t-Distributed Stochastic Neighbor Embedding) from scratch without using machine learning libraries like scikit-learn. Instead, only NumPy, Pandas, Matplotlib, and SVD are used.

Project Overview

The main objective is to reduce 784-dimensional Fashion MNIST data into 2D space while preserving the local structure of data points. The project also explores hyperparameter tuning to optimize the quality of embeddings.

Fashion MNIST Dataset:

Contains 10,000 rows and 785 columns (first column is the label).
Each image has 28 × 28 = 784 pixels, flattened into a single row.
Labels represent 10 different classes of clothing items.

Implementation Workflow

1. Building t-SNE from Scratch
   
i. Implement pairwise similarities in high-dimensional space using the Gaussian function

​ii. Compute pairwise similarities in low-dimensional space using the Student-t distribution

​iii. Optimize the Kullback-Leibler divergence loss

iv. Compute gradients to update the low-dimensional embeddings

v. Update embeddings using momentum-based gradient descent 

2. Hyperparameter Tuning and Evaluation
Run t-SNE with different hyperparameters:
Learning rate (λ),
Momentum (α),
Iteration count,
Perplexity

Evaluate embeddings using:
Sum of centroid distances (D),
Objective function (J)

3. Visualization
Reduce 784 dimensions to 2D and plot the results.
Generate 5 different plots corresponding to different hyperparameter settings.
Use different colors for each label (10 classes).

Results

Comparison of different t-SNE configurations using distance (D) and objective function (J).
Visualization of 10-class Fashion MNIST dataset in 2D space.
Effect of hyperparameter tuning on embedding quality.
