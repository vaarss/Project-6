import csv
import numpy as np

class PCA:
    def __init__(self):
        self.swiss_array = self.csv_reader("swiss_data.csv")

    def transform(self, x):
        x = x - np.mean(self.swiss_array, 0)
        return np.matmul(np.transpose(F), x)

    def csv_reader(self, file):
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
        return csv_reader

if __name__ == '__main__':
    pca = PCA()
    pca.csv_reader("swiss_data.csv")