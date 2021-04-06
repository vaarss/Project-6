""" isomap file """

import numpy as np
from sklearn.utils.graph import graph_shortest_path
import scipy.sparse.linalg
import matplotlib.pyplot as plt

class Isomap:
    """ Isomap constructor """
    def __init__(self, data_points, k):
        """ constructor """
        self.data_points = data_points
        self.k = k

    def euclidian_distances(self, data_points):
        """Create euclidian distances matrix """
        squared_matrix = np.square(data_points)
        sum_squared = np.sum(squared_matrix, axis=1, keepdims=True)
        W = -2 * data_points @ data_points.transpose()
        d_2 = sum_squared.transpose() + sum_squared + W
        distance_matrix = np.sqrt(np.abs(d_2))
        return distance_matrix

    def keep_k_neareset(self, distance_matrix, k):
        """ Only keep the k nearest values """
        holder = distance_matrix
        sorted_indexes = holder.argsort(axis=1)
        sorted_indexes = np.delete(sorted_indexes, slice(k + 1), 1)
        y_dim = distance_matrix.shape[1]
        if k < y_dim:
            for i in range(y_dim):
                distance_matrix[i][sorted_indexes[i]] = 0
        return distance_matrix

    def mds(self, geodesic):
        """ Multidimensional scaling from the taskpaper"""
        d = np.square(geodesic)
        number_of_entries = d.shape[0]
        identity_matrix = np.identity(number_of_entries)
        column_ones = np.ones((number_of_entries, 1))
        # j er centering matrix, denne er riktig
        j = identity_matrix - ((1/number_of_entries)*(column_ones @ column_ones.T))

        b = (-0.5) *j @ d @ j
        # Eigenvalues are sorted
        eigenvalues, eigenvectors = np.linalg.eig(b)  # FØLER eigh er feil, skrev eig istden

        sorted_indexes = eigenvalues.argsort()[::-1]
        eigenvalues = np.real(eigenvalues[sorted_indexes][:2])
        eigenvectors = (eigenvectors.transpose()[sorted_indexes]).transpose()
        eigenvectors = np.real(eigenvectors[:, :2])
        a = np.zeros((2, 2))
        np.fill_diagonal(a, eigenvalues)
        a_power_of_0_point_5 = scipy.linalg.fractional_matrix_power(a, 0.5)
        Y = eigenvectors @ a_power_of_0_point_5
        return Y

    def iso(self):
        """ Method to run all needed methods """
        distance_matrix = self.euclidian_distances(self.data_points)
        distance_matrix = self.keep_k_neareset(distance_matrix, self.k)
        shortest_path = graph_shortest_path(distance_matrix)
        converted_matrix = self.mds(shortest_path)
        return converted_matrix

if __name__ == '__main__':
    inputFiles = ["swiss_data.csv", "digits.csv"]
    INPUT_INDEX = 0

    datapoints = np.genfromtxt((inputFiles[INPUT_INDEX]), delimiter = ',')
    iso = Isomap(datapoints, 30)
    Y = iso.iso()
    if INPUT_INDEX == 0:
        C = np.arange(Y.shape[0])
    else:
        C=np.genfromtxt('digits_label.csv')
    plt.scatter(Y[:,0], Y[:,1], s=10, c=C, cmap="jet")
    plt.show()
