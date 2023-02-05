#### t-SNE

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique that maps high-dimensional data into a lower-dimensional space. t-SNE works by preserving the local structure of the data, so that similar data points are mapped close to each other in the lower-dimensional space. Unlike PCA, t-SNE does not rely on finding a set of uncorrelated variables and is instead based on the probability distribution of the distances between data points in the high-dimensional and low-dimensional spaces. t-SNE is particularly useful when the data has complex non-linear structure, as it can capture these non-linear relationships in the lower-dimensional space.

It is based on a probability distribution over pairs of data points in the high-dimensional space and seeks to minimize the divergence between the distribution in the original space and the distribution in the low-dimensional space. The algorithm works by computing a similarity measure between each pair of points in the high-dimensional space, and then computing a probability distribution over the points in the low-dimensional space that best preserves the similarities between the points in the high-dimensional space. The similarity measure is computed using a Gaussian kernel with a parameter, σ, that controls the width of the kernel.

The similarity measure is given by:
```
D(i, j) = (1 + ||x_i - x_j||^2)^-1/(2*σ^2)
```
Where D(i, j) is the similarity between data points x_i and x_j, and σ is a parameter that determines the amount of smoothing.