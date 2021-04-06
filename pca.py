""" pca file """
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg

class PCA:
    """ PCA class """
    def fit(self, datapoints, dimension_in, dimension_out):
        """ fit method """
        #Center datapoints
        datapoints = datapoints -  datapoints.mean(axis=0, keepdims=True)
        matrix = np.cov(np.transpose(datapoints))
        if dimension_in - 1 > dimension_out:
            [eigenvalues, eigenvectors] = scipy.sparse.linalg.eigs(matrix)
        elif dimension_in -1 == dimension_out: 
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        sorted_indexes = eigenvalues.argsort()
        eigenvectors=eigenvectors[:, sorted_indexes[-dimension_out:]]
        return eigenvectors


    def transform(self, x, datapoints, dimension_in, dimension_out):
        """ Transform from tasksheet"""
        x = x - np.mean(datapoints, axis=0)
        transformation_matrix = self.fit(datapoints, dimension_in, dimension_out)
        return_value = np.transpose(transformation_matrix) @ x
        return return_value[np.newaxis]


    def transform_matrix(self, datapoints):
        """ Transform every datapoint """
        arr = np.zeros((0,2))
        dimension_in = datapoints.shape[1]
        for i in datapoints:
            arr = np.append(arr, self.transform(i, datapoints,dimension_in, 2), axis=0)
        return arr

if __name__ == '__main__':
    """ main """
    pca = PCA()
    inputFiles = ["swiss_data.csv", "digits.csv"]
    inputIndex = 0
    
    datapoints = np.genfromtxt((inputFiles[inputIndex]), delimiter = ',')
    Y = pca.transform_matrix(datapoints)
    if inputIndex == 0:
         C = np.arange(Y.shape[0])
    else:
        C=np.genfromtxt('digits_label.csv')
    plt.scatter(Y[:,0], Y[:,1], s=10, c=C, cmap="jet")
    plt.show()

