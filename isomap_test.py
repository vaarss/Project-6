import numpy as np
from sklearn.utils.graph import graph_shortest_path
import scipy.sparse.linalg
import matplotlib.pyplot as plt

class Isomap:

    def __init__(self, data_points, k): 
        self.data_points = data_points
        self.k = k
        self.iso(self.data_points, self.k)
        
    def euclidian_distances(self, data_points):
        x = np.square(data_points)
        v = np.sum(x, axis=1, keepdims=True)
        W = -2 * data_points @ data_points.transpose()
        d_2 = v.transpose() + v + W
        d = np.sqrt(np.abs(d_2))
        return d

    def keep_k_neareset(self, distance_matrix, k):
        """ Only keep the k nearest values """
        distance_matrix.sort(axis=1)
        y_dim = distance_matrix.shape[0]
        zero_array = np.array([x for x in range(k, y_dim)])
        if len(zero_array) > 0:
            for i in range(y_dim):
                distance_matrix[i][zero_array] = 0
        return distance_matrix

    def mds(self, geodesic):
        geodesic = np.square(geodesic)
        n = geodesic.shape[0]
        identity_matrix = np.identity(n)
        column_ones = np.ones((n, 1))
        j = identity_matrix - ((1/n)*(column_ones @ column_ones.T))
        b = (-0.5) *j @ geodesic @ j
        b = np.abs(b)
        eigenvalues, eigenvectors = np.linalg.eig(b)
        index_of_two_largset = eigenvalues.argsort()[-2:][::-1]
        eigenvectors = eigenvectors[index_of_two_largset]
        eigenvalues = eigenvalues[index_of_two_largset]
        a = np.zeros((2, 2))
        np.fill_diagonal(a, eigenvalues)
        Y = eigenvectors.T @ a**(0.5)
        return Y

    def iso(self, data_points, k):
        distance_matrix = self.euclidian_distances(data_points)
        distance_matrix = self.keep_k_neareset(distance_matrix, k)
        shortest_path = graph_shortest_path(distance_matrix)
        mapped_points = self.mds(shortest_path)
        """ add x and y values for mapping"""
        x = []
        y = []
        for row in mapped_points:
            x.append(row[0])
            y.append(row[1])
        plt.scatter(x,y, c=np.arange(len(x)))
        plt.show()




if __name__ == '__main__':
    datapoints = np.genfromtxt(("digits.csv"), delimiter = ',')
    iso = Isomap(datapoints, 3)