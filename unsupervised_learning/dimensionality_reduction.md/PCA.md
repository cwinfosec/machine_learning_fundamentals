#### PCA

PCA (Principal Component Analysis) is a linear dimensionality reduction technique that transforms the high-dimensional data into a lower-dimensional representation while retaining the most important information. PCA works by finding the directions in which the data varies the most and projects the data onto these directions to form a new set of features. These features are called principal components and they are uncorrelated and ranked in order of their explained variance.

Principal Component Analysis (PCA) is a popular technique in machine learning for dimensionality reduction. The idea behind PCA is to project the original features onto a new set of linearly uncorrelated variables (principal components) that capture the most variance in the data.

The math behind PCA involves linear algebra and eigenvalue decomposition. To perform PCA, we start by computing the covariance matrix of the features, which is a matrix that describes the relationships between all the features in the data. Then, we compute the eigenvalues and eigenvectors of the covariance matrix. The eigenvectors corresponding to the largest eigenvalues are the principal components, and we use these components to project the original data onto a lower-dimensional space.

The formula for PCA is based on finding the eigenvectors and eigenvalues of the covariance matrix of the data. Given a data matrix X with shape (m, n) where m is the number of data points and n is the number of features, the covariance matrix is given by:
```
S = (1 / (m-1)) * X^T * X
```
Where `X^T` represents the transpose of `X`.

The eigenvectors and eigenvalues of `S` represent the directions and magnitudes of the principal components of the data. To perform PCA, we first compute the eigenvectors and eigenvalues of the covariance matrix S, then select the `k` eigenvectors that correspond to the `k` largest eigenvalues. These eigenvectors are then used to project the data onto a lower-dimensional space. The transformed data is obtained by computing the dot product of `X` and the eigenvectors.

The equation for PCA can be expressed as:
```
X_transformed = X * eigenvectors
```
Where `X_transformed` is the transformed data matrix, `X` is the original data matrix, and `eigenvectors` are the eigenvectors of the covariance matrix `S`.

In PCA, the goal is to find the principal components that capture the most variance in the data, so we typically only keep the first few columns of W and use these to transform the data X into X', which is a lower-dimensional representation of the original data.