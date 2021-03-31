import matplotlib
import numpy 
import scipy

class PCA:

    def __init__(self):
        pass

    def fit(self, datapoints, dimension_in, dimension_out): 
        _sum = 0
        centered_data = []
        for i in datapoints: 
            _sum += i
        for i in datapoints: 
            centered_datapoint = i - (_sum/datapoints.size())
            centered_data + centered_datapoint
        matrix = numpy.cov(centered_data)

        if dimension_in - 1 > dimension_out:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(matrix)
        elif dimension_in -1 == dimension_out: 
            eigenvalues, eigenvectors = numpy.linalg.eigh(matrix)
        
        sorted_eigenvalues_index = numpy.argsort(eigenvalues)[-dimension_out:][::-1]
        transformation_matrix = []
        for n in sorted_eigenvalues_index: 
            transformation_matrix[n] = eigenvectors[n]

        return transformation_matrix


        