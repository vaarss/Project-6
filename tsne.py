import csv
import matplotlib.pyplot as plt
import numpy
import random

class tSNE: 

    def __init__(self, datapoints, iterations, k):
        self.iterations = iterations
        self.datapoints = datapoints
        self.k = k
        #self.a = a
        #self.e = e

    def euclidean_distances(self, X):
        """ Finner de parvise euklidiske avstandene mellom datapunktene """
        V = numpy.sum(X * X, axis=1, keepdims=True)
        euclidean_distances = numpy.sqrt(numpy.abs(V.T + V - 2 * (X @ X.T))) 
        print("euklidiske avstander: ", euclidean_distances)
        return euclidean_distances

    def k_nearest_neighbors(self, data): 
        """ Returnerer similarity matrix med 0 og 1 for de k nærmeste naboene """
        k = self.k
        n_data = data.shape[0] 
        dist = numpy.sqrt(self.euclidean_distances(data)**2)
        print("sqrt of squared dist: ", dist)

        indices = dist.argsort()
        neighbors = indices[:, 1:k + 1]
        knn = numpy.zeros((n_data, n_data))
        for i in range(n_data):
            knn[i, neighbors[i, :]] = 1
        print("KNN: ", knn)

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

    """ This is an iterative process. At the beginning you initialize Y from Normal 
    distribution in Step 1 of Section 6.2. Then you update Y in each iteration, where you calculate
     q and Q using the current Y."""

    def t_sne(self, X): 

        similarity_matrix = self.k_nearest_neighbors(X)
        n_data = similarity_matrix.shape[0]

        y = numpy.zeros((n_data,2), float) #y skal være n x 2 (n=64), burde bruke random for  å sette matrisen, kan bruke veldig små verdier
        for row in y: 
            for column in row: 
                column = random.randrange(0, 1)
        print("Y: ", y)
        #bruker y og p(similarity matrix) til å regne ut P og Q 

        q = self.similarity_matrix(X) #bruker y for å lage q

        P = p / numpy.sum(p)
        Q = q / sum(q)
        

        gain = numpy.ones()
        for n in range(self.iterations):
            # Lower momentum the first iterations
            if n < 250: 
                a = 0.5
            else: 
                a = self.a

            q[range(int(len(q))), range(int(len(q)))] = 0  # Set the diagonal to 0, assumes the matrix is square
            Q = q / numpy.sum(q)

            # Add "lying" factor the first 100 iterations
            G = (P - Q) * q if n > 100 else (4 * P - Q) * q
            S = numpy.diag(numpy.sum(G, axis=1))
            grad = 4 * (S - G) @ y

            gain[numpy.sign(grad) == numpy.sign(gain)] *= 0.8
            gain[numpy.sign(grad) != numpy.sign(gain)] += 0.2
            gain[gain < 0.01] = 0.01

            delta = (a * delta) - (self.e * gain * grad)
            y += delta
            

if __name__ == '__main__':
    datapoints = numpy.genfromtxt(("digits.csv"), delimiter = ',')
    tsne = tSNE(datapoints, 100, 2)  
    x = []
    y = []    
    tsne.k_nearest_neighbors(datapoints)
    plt.scatter(x,y)
    plt.show()