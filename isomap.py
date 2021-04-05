import numpy as np
class Isomap:

    def __init__(self, data_points, k): 
        self.data_points = data_points
        self.k = k
        self.euclidian_distances()
        
    def euclidian_distances(self):
        N, D = self.data_points.shape
        euclidian_matrix = np.zeros((N, N))
        for i in self.data_points:
            pass

    def euclidian_distance(self, point_i, point_j, d):
        summer = 0
        for i in range(d):
            summer += (point_i[i]-point_j[i])**2


if __name__ == '__main__':
    datapoints = np.genfromtxt(("swiss_data.csv"), delimiter = ',')
    iso = Isomap(datapoints, 10)