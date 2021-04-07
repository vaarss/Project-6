""" this file holds the tSNE class """
import csv
import matplotlib.pyplot as plt
import numpy
import random

class tSNE:
    """ This class tranforms high dimensional data to 2D data using the tSNE algorithm """
    def __init__(self, datapoints, iterations, k, a):
        """" Initialize tSNE """
        self.iterations = iterations
        self.datapoints = datapoints
        self.k = k
        self.a = a

    def euclidean_distances(self, X):
        """ Finner de parvise euklidske avstandene mellom datapunktene """
        V = numpy.sum(X * X, axis=1, keepdims=True)
        euclidean_distances = numpy.sqrt(numpy.abs(V.T + V - 2 * (X @ X.T))) 
        return euclidean_distances

    def k_nearest_neighbors(self, data):
        """ Returnerer similarity matrix med 0 og 1 for de k nærmeste naboene """
        k = self.k
        n_data = data.shape[0]
        dist = numpy.sqrt(self.euclidean_distances(data)**2)

        indices = dist.argsort()
        neighbors = indices[:, 1:k + 1]
        knn = numpy.zeros((n_data, n_data))
        for i in range(n_data):
            knn[i, neighbors[i, :]] = 1

        #Test for å sjekke at knn inneholder riktig antall 1-ere:
        """
        count_of_ones = 0
        for i in knn:
            for j in i:
                if j == 1:
                    count_of_ones += 1
        print("n_data: ", n_data)
        print("count of ones: ", count_of_ones)
        print("skal være lik n_data: ", count_of_ones/k)
        """
        return knn

    def t_sne(self, X):
        """ method to compute tsne algorithm with optimizations """
        knn = self.k_nearest_neighbors(X)
        p = numpy.maximum(knn, knn.transpose()) #symmetrize knn to get p
        n_data = p.shape[0]
        P = p / numpy.sum(p)
        y = numpy.random.normal(0, 1e-4, (n_data, 2))
        # Skal bruke y og p(similarity matrix) til å regne ut P og Q
        gain = numpy.ones((n_data, 2))
        delta = numpy.zeros((n_data, 2))
        e = 500
        # Loop for iterations:
        for n in range(self.iterations):
            # Lower momentum the first iterations
            if n < 250:
                a = 0.5
            else: 
                a = self.a
            q = 1 / (1 + (numpy.abs(self.euclidean_distances(y)))**2)
            q[range(n_data), range(n_data)] = 0  # Set the diagonal to 0
            Q = q / numpy.sum(q)
            # Add "lying" factor to the first 100 iterations 
            if n <= 100:
                G = (4 * P - Q) * q
            else: 
                G = (P - Q) * q 
            S = numpy.diag(numpy.sum(G, axis=1))
            grad = 4 * (S - G) @ y

            gain[numpy.sign(grad) == numpy.sign(delta)] *= 0.8
            gain[numpy.sign(grad) != numpy.sign(delta)] += 0.2
            gain[gain < 0.01] = 0.01
            delta = (a * delta) - (e * gain * grad)
            y += delta
        return y

if __name__ == '__main__':
    datapoints = numpy.genfromtxt(("digits.csv"), delimiter = ',')
    tsne = tSNE(datapoints, 100, 10, 0.8)
    y = tsne.t_sne(datapoints)
    color = numpy.genfromtxt('digits_label.csv', delimiter='\n')
    plt.scatter(y[:, 0], y[:, 1], s=10, c=color, cmap='jet', marker=".")
    plt.show()
