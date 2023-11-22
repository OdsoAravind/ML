import numpy as np

# Define the initial cluster centers
u1= np.array([6.2, 3.2])
u2= np.array([6.6, 3.7])
u3 = np.array([6.5, 3.0])

# Define the data points in your dataset as a NumPy array 
data_points = np.array([
[5.9, 3.2],
[4.6,2.9],
[6.2,2.8],
[4.7,3.2],
[5.5,4.2],
[5.0,3.0],
[4.9,3.1],
[6.7,3.1],
[5.1,3.8],
[6.0,3.0]])


# Number of clusters (k)
k = 3

# Maximum number of iterations for convergence
max_iterations = 100

# Initialize cluster centers
cluster_centers = np.array([u1, u2, u3])

# Perform k-means clustering
for iteration in range(max_iterations):
# Assign each data point to the nearest cluster
    cluster_assignments = np.argmin(np.linalg.norm(data_points[:, np.newaxis] - cluster_centers, axis=2), axis=1)

# Update cluster centers
    new_centers = np.array([data_points[cluster_assignments == i].mean(axis=0) for i in range(k)])

# Check for convergence
    if np.all(cluster_centers == new_centers):
        break

    cluster_centers = new_centers

# Print the final cluster centers
for i, center in enumerate(cluster_centers):
  print(f"Center of Cluster {i + 1}: {center.round(3)}")

# Print the number of iterations required for convergence
print(f"Number of iterations for convergence: {iteration + 1}")