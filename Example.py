import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd

COLORS = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]

def show_clusters_centroids(clusters,centroids,title,x_var_indx=0,y_var_indx=1,x_var_name='Variable 1',y_var_name="Variable 2",keep=False):

    """
    Show the current clustering for 1 second and save the plot
    Input:
        clusters (list of lists of lists): A List of Clusters. Each cluster
        is also a list of points in the cluster. SEE: k_means.get_clusters()
        centroids (list of lists): A list with the current centroids
        title (string): The title for the plot.
    """

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(
            cluster[:,x_var_indx],
            cluster[:,y_var_indx],
            c = COLORS[i],
            label="Cluster {}".format(i)
        )

    for i, centroid in enumerate(centroids):
        plt.scatter(
            centroid[x_var_indx],
            centroid[y_var_indx],
            c = COLORS[i],
            marker='x',
            s=100
        )

    plt.title(title)
    plt.xlabel(x_var_name)
    plt.ylabel(y_var_name)
    plt.legend()

    if not keep:
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    else:
        plt.show()

def load_data(filename):
    """
    Reads a csv file and returns a list of lists
    """
    with open(filename,'r') as fp:
        data = fp.read().split('\n')
    data_new = [f.split(',') for f in data if f != ""]
    data_formatted = []
    for instance in data_new:
        instance_new = []
        for value in instance:
            try:
                instance_new.append(float(value))
            except ValueError:
                instance_new.append(value)
        data_formatted.append(instance_new)
    return data_formatted

def distance(a,b):
    """
    Compute Euclidean Distance Between Two Points
    Input:
        a (list): an n-dimensional list or array
        b (list): an n-dimensional list or array
    Output:
        The Euclidean Distance between vectors a and b
    """
    return np.sqrt(np.sum((np.array(b)-np.array(a))**2))

def get_clusters(points,centroids):
    """
    Returns a list of clusters given all the points in the dataset and
    the current centroids.
    Input:
        points (list of lists): A list with every point in the dataset
        centroids (list of lists): A list with the current centroids
    Output:
        clusters (list of lists of lists): A List of Clusters. Each cluster
        is also a list of points in the cluster.
    """
    clusters = [[] for f in centroids]

    for i, point in enumerate(points):
        point_to_centroids = []
        for j, centroid in enumerate(centroids):
            point_to_centroids.append(distance(point,centroid))
        closest_idx = np.argmin(point_to_centroids)
        clusters[closest_idx].append(point)

    return clusters

def update_centroids(clusters):
    """
    Given a list of clusters (as prepared by get_clusters) get the new centroids
    Input:
        clusters (list of lists of lists): A List of Clusters. Each cluster
        is also a list of points in the cluster.
    Output:
        A (list of lists): The new centroids.
    """
    new_centroids = []

    for cluster in clusters:
        new_centroids.append(np.mean(cluster,axis = 0))
    return new_centroids



def k_means(points, k, iterations=10):
    """
    K Means Unsupervised ML Algorithm Implementation with Forgy Initialization
    Input:
        points (numpy array): a 2D Array with the data to cluster.
        k (int): The number of clusters to find
    """
    idx = np.random.randint(len(points),size=k)

    centroids = points[idx,:]
    clusters = get_clusters(points,centroids)

    for i in range(iterations):

        if i % 1 == 0:
            if i == 0:
                title = "Initialization"
            else:
                title = "Iteration {}".format(i+1)

            show_clusters_centroids(
                clusters,
                centroids,
                title,
            )

        clusters = get_clusters(points,centroids)
        centroids = update_centroids(clusters)

    return clusters,centroids


if __name__ == "__main__":
    data = load_data('./data/datasets_14701_19663_CC GENERAL.csv')
    k = 3

    X = np.array([f[:-1] for f in data])
    y = np.array([f[-1] for f in data])

    clusters,centroids = k_means(X,3)

    show_clusters_centroids(clusters,centroids,"Result", keep=True)
    plt.show()