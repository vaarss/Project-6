import csv
import matplotlib.pyplot as plt
import numpy
import scipy.sparse.linalg

class PCA:
    def __init__(self):
        pass

    def fit(self, datapoints, dimension_in, dimension_out):
        datapoints = datapoints -  datapoints.mean(axis=0)
        matrix = numpy.cov(numpy.transpose(datapoints))
        if dimension_in - 1 > dimension_out:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(matrix)
        elif dimension_in -1 == dimension_out: 
            eigenvalues, eigenvectors = numpy.linalg.eigh(matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors=eigenvectors[:, idx[-dimension_out:]]
        eigenvectors = numpy.array(eigenvectors)
        return eigenvectors

    def transform(self, x, datapoints):
        x = x - numpy.mean(datapoints, axis=0)
        dimension_in = len(x)
        transformation_matrix = self.fit(datapoints, dimension_in, 2)
        return_value = numpy.matmul(numpy.transpose(transformation_matrix), x)
        return return_value[numpy.newaxis]


    def transform_matrix(self, datapoints):
        arr = numpy.zeros((0,2))
        for i in datapoints:
            arr = numpy.append(arr, self.transform(i, datapoints), axis=0)
        return arr

if __name__ == '__main__':
    pca = PCA()
    datapoints = numpy.genfromtxt(("digits.csv"), delimiter = ',')
    transformed = pca.transform_matrix(datapoints)
    x = []
    y = []
    for data_point in transformed:
        x.append(data_point[0])
        y.append(data_point[1])
    plt.scatter(x,y)
    plt.show()