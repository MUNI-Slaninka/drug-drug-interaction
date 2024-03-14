from matplotlib.pyplot import *
import csv


def load_csv(filename, data_type):  # load csv, ignore the first row,type=int, data read as int， else float
    matrix_data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row_vector in csvreader:
            if data_type == 'int':
                matrix_data.append(list(map(int, row_vector[1:])))
            else:
                matrix_data.append(list(map(float, row_vector[1:])))
    return np.matrix(matrix_data)





