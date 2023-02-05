#### K-means

K-means clustering is an unsupervised learning algorithm that is used to partition data into ‘k’ clusters. The algorithm works by first randomly selecting ‘k’ points as cluster centers and then assigning each data point to its closest cluster center. The algorithm then iteratively adjusts the cluster centers to better fit the data points until the cluster centers no longer move.

K-means clustering is a useful tool for data exploration and finding underlying patterns in data. It can be used to group similar data points together, allowing for more efficient analysis of the data.

The math behind k-means clustering is based on the concept of minimizing the sum of squared distances between points and their assigned cluster centroid. The goal of k-means is to divide a set of points into k clusters, where each cluster is represented by its centroid.

The k-means algorithm starts by randomly selecting k initial centroids. Then, it repeatedly performs two steps until convergence:

Assignment: Each point is assigned to the closest centroid based on the Euclidean distance.
Recalculation: The centroids are updated by computing the mean of the points assigned to each cluster.
The algorithm continues to iterate until the centroids stop changing or a maximum number of iterations is reached.

The optimization objective of k-means can be mathematically expressed as the following formula:
```
J(C) = 1/n ∑_{i=1}^k ∑_{x ∈ C_i} ||x - μ_i||^2
```
where `J(C)` is the sum of squared distances, `n` is the number of points, `C_i` is the set of points assigned to cluster `i`, and `μ_i` is the centroid of cluster `i`. The goal is to find the values of `C` that minimize `J(C)`.

A simple use-case example of k-means clustering would be a retail store looking to segment their customer base into different groups. The store could use k-means clustering to group customers based on their purchase history, age, and other demographic information. This would allow the store to better target their marketing efforts and tailor their product offerings to each customer segment.

Here is a simple example of using k-means clustering in Python to segment a retail store's customer base:
```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the customer data from a CSV file
customers = pd.read_csv("customers.csv")

# Extract the relevant features for clustering (age, location, and purchase history)
X = customers[["age", "location", "purchase_history"]].values

# Fit the k-means model to the data
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Get the cluster labels for each customer
labels = kmeans.labels_

# Add the cluster labels to the customer data
customers["cluster"] = labels

# Calculate the size of each cluster
cluster_sizes = customers.groupby("cluster").size().reset_index(name="size")

# Calculate the mean age, location, and purchase history for each cluster
cluster_stats = customers.groupby("cluster").mean().reset_index()

# Print the results
print("Cluster Sizes:")
print(cluster_sizes)
print("\nCluster Statistics:")
print(cluster_stats)

# Plot the data to visualize the clusters
plt.scatter(customers["age"], customers["purchase_history"], c=labels)
plt.xlabel("Age")
plt.ylabel("Purchase History")
plt.show()
```

In this example, the customer data is loaded from a CSV file customers.csv and stored in a pandas DataFrame. The relevant features for clustering (age, location, and purchase history) are extracted and stored in the X numpy array. The k-means model is fit to the data using the KMeans class from the scikit-learn library, with n_clusters=3 to segment the customers into 3 distinct groups. The cluster labels are calculated using kmeans.labels_ and added to the customer data as a new column. The size of each cluster and the mean age, location, and purchase history for each cluster are calculated using pandas' groupby and size and mean functions. Finally, the results are printed and the data is plotted using matplotlib to visualize the clusters.

The `customers.csv` file should have columns for the customer's `age`, `location`, and `purchase history`. For each customer, the purchase history column would contain values indicating the products they have purchased. For example, if the customer purchased apples and juice, the value for the purchase history column could be "apples, juice". The data in the file could look like this:
```
Age,Location,Purchase History
25,New York,apples
35,California,juice
45,Texas,steak
30,Illinois,apples, juice
40,Florida,steak, apples
50,Arizona,juice, steak
```

This is just one example of how the file could be formatted to contain data for the three products apples, juice, and steak. You could choose to encode the data differently, such as using 0/1 values to indicate the presence or absence of each product, but the general structure of the file would remain the same.