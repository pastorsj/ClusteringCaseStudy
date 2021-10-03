from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Nicely clustered set of values
# https://pythonprogramminglanguage.com/kmeans-elbow-method/
clustered_x_values = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
clustered_y_values = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

unclustered_x_values = np.array([random.random() * 10 for _ in range(1000)])
unclustered_y_values = np.array([random.random() * 10 for _ in range(1000)])

# List of K-Clusters
k_range = range(1, 10)


def create_points(x_values, y_values):
    return np.array(list(zip(x_values, y_values))).reshape(len(x_values), 2)


def run_elbow_method_using_distortion(x_values, y_values):
    list_of_distortions = []
    points = create_points(x_values, y_values)
    for k in k_range:
        k_means = KMeans(n_clusters=k)
        k_means.fit(points)

        # Calculate distortion per k value
        centroids = k_means.cluster_centers_
        distances_from_centroids = cdist(points, centroids, 'euclidean')
        smallest_distances_from_centroids = np.min(distances_from_centroids, axis=1)
        calculated_distortion = sum(smallest_distances_from_centroids) / points.shape[0]

        list_of_distortions.append(calculated_distortion)

    plot_elbow_method(k_range, list_of_distortions, 'Distortion', 'Plot of elbow method using Distortion')


def run_elbow_method_using_inertia(x_values, y_values):
    list_of_inertias = []
    points = create_points(x_values, y_values)
    for k in k_range:
        k_means = KMeans(n_clusters=k)
        k_means.fit(points)

        # Calculate inertia per k value
        calculated_inertia = k_means.inertia_  # Sum of the squared distances to the nearest cluster

        list_of_inertias.append(calculated_inertia)

    plot_elbow_method(k_range, list_of_inertias, 'Inertia', 'Plot of elbow method using Inertia')


def plot_points(x_values, y_values):
    plt.plot()
    plt.scatter(x_values, y_values)
    plt.title('Plot of the dataset')
    plt.show()


def plot_elbow_method(k_values, y_values, y_label, title):
    plt.plot(k_values, y_values, 'bx-')
    plt.xlabel('k')
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    print('Plotting the elbow method using predefined clusters')
    plot_points(clustered_x_values, clustered_y_values)
    run_elbow_method_using_distortion(clustered_x_values, clustered_y_values)
    run_elbow_method_using_inertia(clustered_x_values, clustered_y_values)

    print('Now for random clusters')
    plot_points(unclustered_x_values, unclustered_y_values)
    run_elbow_method_using_distortion(unclustered_x_values, unclustered_y_values)
    run_elbow_method_using_inertia(unclustered_x_values, unclustered_y_values)

