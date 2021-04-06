import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg

class PCA:
    def __init__(self):
        pass


    def fit(self, datapoints, dimension_in, dimension_out):
        datapoints = datapoints -  datapoints.mean(axis=0)
        matrix = np.cov(np.transpose(datapoints))
        if dimension_in - 1 > dimension_out:
            [eigenvalues, eigenvectors] = scipy.sparse.linalg.eigs(matrix)
        elif dimension_in -1 == dimension_out: 
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors=eigenvectors[:, idx[-dimension_out:]]
        eigenvectors = np.array(eigenvectors)
        return eigenvectors


    def transform(self, x, datapoints):
        x = x - np.mean(datapoints, axis=0)
        dimension_in = len(x)
        transformation_matrix = self.fit(datapoints, dimension_in, 2)
        return_value = np.matmul(np.transpose(transformation_matrix), x)
        return return_value[np.newaxis]


    def transform_matrix(self, datapoints):
        arr = np.zeros((0,2))
        for i in datapoints:
            arr = np.append(arr, self.transform(i, datapoints), axis=0)
        return arr

if __name__ == '__main__':
    pca = PCA()
    datapoints = np.genfromtxt(("swiss_data.csv"), delimiter = ',')
    transformed = pca.transform_matrix(datapoints)
    x = []
    y = []
    for data_point in transformed:
        x.append(data_point[0])
        y.append(data_point[1])
    N = datapoints.shape[0]
    plt.scatter(x,y, c=np.arange(N))
    plt.show()

