import csv
import matplotlib
import numpy 
import scipy

class PCA:
    def __init__(self):
        pass

    def fit(self, datapoints, dimension_in, dimension_out): 
        _sum = 0
        centered_data = []
        my = numpy.mean(datapoints, axis=0)
        print("my: ", my)
        print("datapoints", datapoints)
        for i in datapoints: 
            centered_data.append(i - my)
        print("centered data: ", centered_data)
        matrix = numpy.cov(centered_data)
        print("cov matrix: ", matrix)
        if dimension_in - 1 > dimension_out:
            [eigenvalues, eigenvectors] = scipy.sparse.linalg.eigs(matrix)
        elif dimension_in -1 == dimension_out: 
            [eigenvalues, eigenvectors] = numpy.linalg.eigh(matrix)
        print("Eigenvectors", eigenvectors)
        sorted_eigenvalues_index = numpy.argsort(eigenvalues)[-dimension_out:][::-1]
        transformation_matrix = []
        for n in sorted_eigenvalues_index:
            print("eigenvectors[n] (skal printes 2 ganger): ", eigenvectors[n])
            transformation_matrix.append(eigenvectors[n])
        print("transformation matrix: ", transformation_matrix)
        return transformation_matrix

    def transform(self, x, datapoints):
        x = x - numpy.mean(datapoints, axis=0)
        print("x in transform: ", x)
        dimension_in = len(x)
        transformation_matrix = self.fit(datapoints, dimension_in, 2)
        print("x: ", x)
        print("transposed: ", numpy.transpose(transformation_matrix))
        return numpy.matmul(numpy.transpose(transformation_matrix), x)

    def csv_reader(self, file):
        with open(file) as csv_file:
            csv_reader = list(csv.reader(csv_file, delimiter=','))
        desired_array = [[float(numeric_string) for numeric_string in x] for x in csv_reader]
        return desired_array

    def transform_matrix(self, datapoints):
        new_list = []
        for i in datapoints:
            new_list.append(self.transform(i, datapoints))
        print("new list: ", new_list)
        return new_list

if __name__ == '__main__':
    pca = PCA()
    datapoints = pca.csv_reader("swiss_data.csv")
    pca.transform_matrix(datapoints)